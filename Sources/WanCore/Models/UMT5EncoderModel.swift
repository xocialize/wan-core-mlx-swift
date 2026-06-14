//
//  UMT5EncoderModel.swift
//
//  LIFTED from the Swift component donor:
//    /Volumes/DEV_ARCHIVE/longcat-avatar-mlx-swift
//      Sources/LongCatVideoAvatar/Models/UMT5EncoderModel.swift
//  Parity-proven there: 0.119 max_abs vs Python-MLX. Content kept as verbatim
//  as possible; the parity-critical choices (NO 1/sqrt(d) QK^T scaling, fp32
//  softmax, per-block relative position bias / sharedPos=false) are unchanged.
//
//  Cross-checked module-by-module against the Wan2.2 backbone reference
//  `mlx_video/models/wan_2/text_encoder.py` (T5Encoder et al.): bias flags
//  (all Linears bias-free), eps 1e-6, buckets 32, maxDist 128, mask constant
//  -3.389e38, gated-GeLU(tanh) FFN, and the bucket math are identical.
//
//  UMT5-XXL text encoder. Bernini's converted checkpoint ships it as a single
//  flat `t5_encoder.safetensors` (242 tensors, all-weight-no-bias) using the
//  compact MLX names (token_embedding / pos_embedding /
//  blocks.X.attn.{q,k,v,o} / blocks.X.ffn.{gate_proj,fc1,fc2} /
//  blocks.X.norm{1,2} / norm) — the key contract is pinned in
//  `BerniniWeightKeys.t5Keys()`.
//
//  Compared to vanilla T5:
//  - Per-block relative position bias (`shared_pos=false`)
//  - Gated GeLU FFN with `gate_proj` / `fc1` / `fc2` (T5 1.1 style)
//  - RMSNorm (HF still calls it `T5LayerNorm`)
//  - No bias on Linear projections
//  - No 1/sqrt(d) scaling on QK^T; softmax done in fp32
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - T5LayerNorm

/// RMS-based layer normalization (T5/UMT5 style). Routes through
/// `MLXFast.rmsNorm` (analogous to Python's `mx.fast.rms_norm`).
public final class T5LayerNorm: Module, @unchecked Sendable {
    public let eps: Float
    public let weight: MLXArray

    public init(dim: Int, eps: Float = 1e-6) {
        self.eps = eps
        self.weight = MLXArray.ones([dim])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - T5RelativeEmbedding

/// T5-style bucketed relative position bias. For UMT5 (`shared_pos=false`)
/// one instance lives per block.
public final class T5RelativeEmbedding: Module, @unchecked Sendable {
    public let numBuckets: Int
    public let numHeads: Int
    public let bidirectional: Bool
    public let maxDist: Int
    public var embedding: Embedding

    public init(
        numBuckets: Int,
        numHeads: Int,
        bidirectional: Bool = true,
        maxDist: Int = 128
    ) {
        self.numBuckets = numBuckets
        self.numHeads = numHeads
        self.bidirectional = bidirectional
        self.maxDist = maxDist
        self.embedding = Embedding(embeddingCount: numBuckets, dimensions: numHeads)
        super.init()
    }

    private func relativePositionBucket(_ relPos: MLXArray) -> MLXArray {
        var relPos = relPos
        var relBuckets: MLXArray
        let halfBuckets: Int

        if bidirectional {
            halfBuckets = numBuckets / 2
            relBuckets = (relPos .> 0).asType(.int32) * halfBuckets
            relPos = MLX.abs(relPos)
        } else {
            halfBuckets = numBuckets
            relBuckets = MLXArray.zeros(like: relPos).asType(.int32)
            relPos = MLX.maximum(-relPos, MLXArray.zeros(like: relPos))
        }

        let maxExact = halfBuckets / 2
        let isSmall = relPos .< maxExact

        let relPosF = relPos.asType(.float32)
        let logRatio = MLX.log(relPosF / Float(maxExact)) / Float(log(Double(maxDist) / Double(maxExact)))
        var relPosLarge = (MLXArray(Float(maxExact)) + logRatio * Float(halfBuckets - maxExact)).asType(.int32)
        relPosLarge = MLX.minimum(
            relPosLarge,
            MLXArray.full(relPosLarge.shape, values: MLXArray(Int32(halfBuckets - 1)))
        )

        relBuckets = relBuckets + MLX.which(isSmall, relPos.asType(.int32), relPosLarge)
        return relBuckets
    }

    /// Returns `[1, num_heads, lq, lk]`.
    public func callAsFunction(lq: Int, lk: Int) -> MLXArray {
        let positionsK = MLXArray(0..<lk).expandedDimensions(axis: 0)
        let positionsQ = MLXArray(0..<lq).expandedDimensions(axis: 1)
        let relPos = positionsK - positionsQ
        let buckets = relativePositionBucket(relPos)
        let embeds = embedding(buckets)
        return embeds.transposed(2, 0, 1).expandedDimensions(axis: 0)
    }
}

// MARK: - T5Attention

/// T5/UMT5 multi-head attention. No 1/sqrt(d) scaling (unscaled QK^T).
/// Softmax in fp32 regardless of the input dtype (per the original T5 paper;
/// fused SDPA in bf16 loses precision because the unscaled logits get large).
public final class T5Attention: Module, @unchecked Sendable {
    public let dim: Int
    public let dimAttn: Int
    public let numHeads: Int
    public let headDim: Int

    public var q: Linear
    public var k: Linear
    public var v: Linear
    public var o: Linear

    public init(dim: Int, dimAttn: Int, numHeads: Int) {
        precondition(dimAttn % numHeads == 0)
        self.dim = dim
        self.dimAttn = dimAttn
        self.numHeads = numHeads
        self.headDim = dimAttn / numHeads

        self.q = Linear(dim, dimAttn, bias: false)
        self.k = Linear(dim, dimAttn, bias: false)
        self.v = Linear(dim, dimAttn, bias: false)
        self.o = Linear(dimAttn, dim, bias: false)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        context: MLXArray? = nil,
        mask: MLXArray? = nil,
        posBias: MLXArray? = nil
    ) -> MLXArray {
        // Reconciled with text_encoder.py: `context` (cross-attention capable,
        // defaults to self-attention). The encoder path never passes it.
        let context = context ?? x
        let b = x.dim(0)
        let n = numHeads
        let c = headDim

        let qx = q(x).reshaped(b, -1, n, c).transposed(0, 2, 1, 3)
        let kx = k(context).reshaped(b, -1, n, c).transposed(0, 2, 1, 3)
        let vx = v(context).reshaped(b, -1, n, c).transposed(0, 2, 1, 3)

        // T5 convention: NO 1/sqrt(d) scaling, softmax in fp32.
        var attn = MLX.matmul(qx.asType(.float32), kx.asType(.float32).transposed(0, 1, 3, 2))
        if let posBias {
            attn = attn + posBias.asType(.float32)
        }
        if let mask {
            var m = mask
            if m.ndim == 2 { m = m[0..., .newAxis, .newAxis, 0...] }
            else if m.ndim == 3 { m = m[0..., .newAxis, 0..., 0...] }
            let additive = MLX.which(m .== 0, MLXArray(Float(-3.389e38)), MLXArray(Float(0)))
                .asType(.float32)
            attn = attn + additive
        }
        let sm = MLX.softmax(attn, axis: -1).asType(qx.dtype)
        let out = MLX.matmul(sm, vx).transposed(0, 2, 1, 3).reshaped(b, -1, n * c)
        return o(out)
    }
}

// MARK: - T5FeedForward

/// Gated GeLU FFN (T5 1.1 / UMT5): `gate_proj` gates `fc1` → `fc2`.
public final class T5FeedForward: Module, @unchecked Sendable {
    public let dim: Int
    public let dimFFN: Int

    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    public var fc1: Linear
    public var fc2: Linear

    public init(dim: Int, dimFFN: Int) {
        self.dim = dim
        self.dimFFN = dimFFN
        self._gateProj.wrappedValue = Linear(dim, dimFFN, bias: false)
        self.fc1 = Linear(dim, dimFFN, bias: false)
        self.fc2 = Linear(dimFFN, dim, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Python uses nn.GELU(approx="tanh") on gate_proj output; the free
        // function `geluApproximate` is the same tanh approximation (and,
        // unlike the Python module attribute, contributes no parameter path).
        fc2(fc1(x) * geluApproximate(gateProj(x)))
    }
}

// MARK: - T5SelfAttentionBlock

/// One UMT5 encoder block: pre-LN self-attn + pre-LN gated FFN. For UMT5
/// the per-block relative position bias lives here (`posEmbedding`); for
/// vanilla T5 (`sharedPos=true`) it's `nil` and the encoder passes in the
/// shared bias.
public final class T5SelfAttentionBlock: Module, @unchecked Sendable {
    public let sharedPos: Bool

    public var norm1: T5LayerNorm
    public var attn: T5Attention
    public var norm2: T5LayerNorm
    public var ffn: T5FeedForward

    @ModuleInfo(key: "pos_embedding") public var posEmbedding: T5RelativeEmbedding?

    public init(
        dim: Int,
        dimAttn: Int,
        dimFFN: Int,
        numHeads: Int,
        numBuckets: Int,
        sharedPos: Bool = true
    ) {
        self.sharedPos = sharedPos
        self.norm1 = T5LayerNorm(dim: dim)
        self.attn = T5Attention(dim: dim, dimAttn: dimAttn, numHeads: numHeads)
        self.norm2 = T5LayerNorm(dim: dim)
        self.ffn = T5FeedForward(dim: dim, dimFFN: dimFFN)
        self._posEmbedding.wrappedValue = sharedPos ? nil :
            T5RelativeEmbedding(numBuckets: numBuckets, numHeads: numHeads, bidirectional: true)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        posBias: MLXArray? = nil
    ) -> MLXArray {
        let e: MLXArray? = sharedPos ? posBias : posEmbedding!(lq: x.dim(1), lk: x.dim(1))
        var y = x + attn(norm1(x), mask: mask, posBias: e)
        y = y + ffn(norm2(y))
        return y
    }
}

// MARK: - UMT5EncoderModel

/// UMT5-XXL encoder (Python reference: `wan_2/text_encoder.py` `T5Encoder`).
public final class UMT5EncoderModel: Module, @unchecked Sendable {
    public let dim: Int
    public let sharedPos: Bool

    @ModuleInfo(key: "token_embedding") public var tokenEmbedding: Embedding
    @ModuleInfo(key: "pos_embedding") public var posEmbedding: T5RelativeEmbedding?
    public var blocks: [T5SelfAttentionBlock]
    public var norm: T5LayerNorm

    public init(
        vocabSize: Int = 256384,
        dim: Int = 4096,
        dimAttn: Int = 4096,
        dimFFN: Int = 10240,
        numHeads: Int = 64,
        numLayers: Int = 24,
        numBuckets: Int = 32,
        sharedPos: Bool = false   // UMT5 default — per-block relative bias
    ) {
        self.dim = dim
        self.sharedPos = sharedPos

        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: dim)
        self._posEmbedding.wrappedValue = sharedPos
            ? T5RelativeEmbedding(numBuckets: numBuckets, numHeads: numHeads, bidirectional: true)
            : nil
        var bs: [T5SelfAttentionBlock] = []
        for _ in 0..<numLayers {
            bs.append(T5SelfAttentionBlock(
                dim: dim, dimAttn: dimAttn, dimFFN: dimFFN,
                numHeads: numHeads, numBuckets: numBuckets,
                sharedPos: sharedPos
            ))
        }
        self.blocks = bs
        self.norm = T5LayerNorm(dim: dim)
        super.init()
    }

    /// Construct from the converted checkpoint's resolved `config.json`
    /// (the donor used a HF-style umT5 config; Bernini's conversion folds the
    /// t5 hyperparameters into the single Wan config at the checkpoint root).
    public static func fromConfig(_ config: WanConfig) -> UMT5EncoderModel {
        UMT5EncoderModel(
            vocabSize: config.t5VocabSize,
            dim: config.t5Dim,
            dimAttn: config.t5DimAttn,
            dimFFN: config.t5DimFfn,
            numHeads: config.t5NumHeads,
            numLayers: config.t5NumLayers,
            numBuckets: config.t5NumBuckets,
            sharedPos: false
        )
    }

    /// Forward pass.
    /// - ids:  `[B, L]` token ids
    /// - mask: `[B, L]` attention mask (1=keep, 0=pad); optional
    /// - Returns: `[B, L, dim]` hidden states
    public func callAsFunction(_ ids: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var x = tokenEmbedding(ids)
        let e = posEmbedding?(lq: x.dim(1), lk: x.dim(1))
        for block in blocks {
            x = block(x, mask: mask, posBias: e)
        }
        x = norm(x)
        return x
    }

    /// Download + load the published flat `t5_encoder.safetensors` into a
    /// fully-initialized model. The key set is verified against the pinned
    /// contract (`BerniniWeightKeys.t5Keys()`, 242 keys) — 0 missing /
    /// 0 unused, or the load throws. The t5 encoder ships bf16 in both
    /// published variants (the int4 recipe quantizes only DiT block Linears).
    public static func fromPretrained(
        _ repoID: String = "mlx-community/Bernini-R-bf16",
        progress: (@Sendable (_ file: String, _ done: Int, _ total: Int) -> Void)? = nil
    ) async throws -> UMT5EncoderModel {
        let root = try await WeightLoader.snapshotDownload(repoID: repoID, progress: progress)
        let config = try WanConfig.load(from: root.appendingPathComponent("config.json"))
        let model = UMT5EncoderModel.fromConfig(config)
        let url = try WeightLoader.componentFile("t5_encoder", under: root)
        var weights = try WeightLoader.loadVerifiedSafetensors(
            url: url,
            expectedKeys: BerniniWeightKeys.t5Keys(layers: config.t5NumLayers)
        )
        // Upcast bf16 checkpoint weights to fp32, matching mlx-video's
        // load_t5_encoder: the encoder runs once per generation, and the
        // official implementation computes its softmax in float32.
        weights = weights.mapValues { $0.asType(.float32) }
        let updated = ModuleParameters.unflattened(weights)
        try model.update(parameters: updated, verify: [.noUnusedKeys])
        return model
    }
}
