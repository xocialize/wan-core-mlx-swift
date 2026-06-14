// 1:1 translation of mlx_video/models/wan_2/wan_2.py (mlx-video @ 87db56a).
// WanModel — the Wan2.2 diffusion backbone (one expert; Bernini-R loads two).
//
// Non-parameter buffers (`freqs`, the time-embedding inv_freq) live in a
// plain holder class so the flattened parameter set is exactly the 1095-key
// contract; the int4 checkpoint's stray serialized `freqs` is dropped by the
// loader's tolerated-extras list.

import Foundation
import MLX
import MLXNN

/// Compute sinusoidal positional embeddings.
/// position: 1D [L] or 2D [B, L] -> [L, dim] or [B, L, dim].
func sinusoidalEmbedding1d(_ dim: Int, _ position: MLXArray) -> MLXArray {
    precondition(dim % 2 == 0)
    let half = dim / 2
    let pos = position.asType(.float32)
    let invFreq = pow(
        MLXArray(Float(10000.0)),
        -MLXArray(0..<half).asType(.float32) / Float(half))
    let sinusoid = pos.expandedDimensions(axis: -1) * invFreq
    return concatenated([cos(sinusoid), sin(sinusoid)], axis: -1)
}

/// Non-parameter constant storage (kept out of Module reflection so these
/// never appear in the weight key contract).
private final class ConstantBuffers {
    let freqs: MLXArray
    let invFreq: MLXArray

    init(headDim: Int, freqDim: Int) {
        // Three rope tables with per-axis dim normalization (t gets the
        // remainder), concatenated along the frequency axis.
        let d = headDim
        self.freqs = concatenated(
            [
                ropeParams(1024, d - 4 * (d / 6)),
                ropeParams(1024, 2 * (d / 6)),
                ropeParams(1024, 2 * (d / 6)),
            ],
            axis: 1
        )
        // Sinusoidal inv_freq computed in float64 (numpy parity), stored fp32.
        let half = freqDim / 2
        self.invFreq = MLXArray(
            (0..<half).map { Float(pow(10000.0, -Double($0) / Double(half))) })
    }
}

/// Output projection head with learned modulation.
public class Head: Module {
    let outDim: Int
    let patchSize: [Int]
    @ModuleInfo(key: "norm") var norm: WanLayerNorm
    @ModuleInfo(key: "head") var head: Linear
    let modulation: MLXArray

    init(dim: Int, outDim: Int, patchSize: [Int], eps: Float = 1e-6) {
        self.outDim = outDim
        self.patchSize = patchSize
        let projDim = patchSize.reduce(1, *) * outDim
        self._norm.wrappedValue = WanLayerNorm(dim, eps)
        self._head.wrappedValue = Linear(dim, projDim)
        self.modulation = (MLXRandom.normal([1, 2, dim]) * pow(Float(dim), -0.5))
            .asType(.float32)
    }

    /// x: [B, L, dim]; e: [B, dim] / [B, 1, dim] (broadcast) / [B, L, dim] (per-token)
    public func callAsFunction(_ x: MLXArray, _ e: MLXArray) -> MLXArray {
        var e = e
        if e.ndim == 2 {
            e = e.expandedDimensions(axis: 1)  // [B, 1, dim]
        }
        // Modulation in float32 (matching reference's autocast(float32))
        let mod = modulation.expandedDimensions(axis: 1) + e.expandedDimensions(axis: 2)
        let e0 = mod[0..., 0..., 0, 0...]  // [B, L_e, dim] shift
        let e1 = mod[0..., 0..., 1, 0...]  // [B, L_e, dim] scale
        let xMod = norm(x) * (1 + e1) + e0
        return head(xMod)
    }
}

/// Text conditioning input: raw per-prompt UMT5 features, or the output of
/// `embedText` reused across denoising steps. (Python overloads one argument.)
public enum WanTextContext {
    case raw([MLXArray])
    case embedded(MLXArray)
}

/// Wan2.2 diffusion backbone for text-to-video generation.
public final class WanModel: Module, @unchecked Sendable {
    let config: WanConfig
    public let dim: Int
    let numHeads: Int
    let outDim: Int
    let patchSize: [Int]
    let textLen: Int
    let freqDim: Int

    @ModuleInfo(key: "patch_embedding_proj") public var patchEmbeddingProj: Linear
    @ModuleInfo(key: "text_embedding_0") var textEmbedding0: Linear
    @ModuleInfo(key: "text_embedding_1") var textEmbedding1: Linear
    @ModuleInfo(key: "time_embedding_0") public var timeEmbedding0: Linear
    @ModuleInfo(key: "time_embedding_1") public var timeEmbedding1: Linear
    @ModuleInfo(key: "time_projection") public var timeProjection: Linear
    @ModuleInfo(key: "blocks") public var blocks: [WanAttentionBlock]
    @ModuleInfo(key: "head") public var head: Head

    private let buffers: ConstantBuffers
    public var freqs: MLXArray { buffers.freqs }
    public var invFreq: MLXArray { buffers.invFreq }

    public init(_ config: WanConfig) {
        self.config = config
        self.dim = config.dim
        self.numHeads = config.numHeads
        self.outDim = config.outDim
        self.patchSize = config.patchSize
        self.textLen = config.textLen
        self.freqDim = config.freqDim

        // Patch embedding: Conv3d implemented as a reshaped linear
        let patchDim = config.inDim * config.patchSize.reduce(1, *)
        self._patchEmbeddingProj.wrappedValue = Linear(patchDim, config.dim)

        // Text embedding MLP (GELU-tanh between the two linears)
        self._textEmbedding0.wrappedValue = Linear(config.textDim, config.dim)
        self._textEmbedding1.wrappedValue = Linear(config.dim, config.dim)

        // Time embedding MLP (SiLU between the two linears)
        self._timeEmbedding0.wrappedValue = Linear(config.freqDim, config.dim)
        self._timeEmbedding1.wrappedValue = Linear(config.dim, config.dim)

        // Time projection for modulation (SiLU then 6x dim)
        self._timeProjection.wrappedValue = Linear(config.dim, config.dim * 6)

        self._blocks.wrappedValue = (0..<config.numLayers).map { _ in
            WanAttentionBlock(
                dim: config.dim,
                ffnDim: config.ffnDim,
                numHeads: config.numHeads,
                windowSize: (config.windowSize[0], config.windowSize[1]),
                qkNorm: config.qkNorm,
                crossAttnNorm: config.crossAttnNorm,
                eps: Float(config.eps)
            )
        }

        self._head.wrappedValue = Head(
            dim: config.dim, outDim: config.outDim, patchSize: config.patchSize,
            eps: Float(config.eps))

        self.buffers = ConstantBuffers(
            headDim: config.dim / config.numHeads, freqDim: config.freqDim)
    }

    /// Convert one video latent [C, F, H, W] to patch embeddings.
    /// Returns (patches [1, L, dim], gridSize (F', H', W')).
    public func patchify(_ x: MLXArray) -> (MLXArray, (Int, Int, Int)) {
        let (c, f, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let (pt, ph, pw) = (patchSize[0], patchSize[1], patchSize[2])

        let fOut = f / pt
        let hOut = h / ph
        let wOut = w / pw

        // [C, F, H, W] -> [F', H', W', C, pt, ph, pw] -> [L, C*pt*ph*pw]
        // Order must be [C, pt, ph, pw] (C slowest) to match Conv3d weight layout
        var x = x.reshaped(c, fOut, pt, hOut, ph, wOut, pw)
        x = x.transposed(1, 3, 5, 0, 2, 4, 6)
        x = x.reshaped(fOut * hOut * wOut, -1)

        // Project and cast to model dtype to prevent float32 cascade from input latents
        var patches = patchEmbeddingProj(x)
        patches = patches.asType(linearDtype(patchEmbeddingProj))
        patches = patches.expandedDimensions(axis: 0)  // [1, L, dim]

        return (patches, (fOut, hOut, wOut))
    }

    /// Reconstruct videos from patch embeddings.
    /// x: [B, L, outDim * prod(patchSize)] -> list of [C, F, H, W].
    public func unpatchify(_ x: MLXArray, gridSizes: [(Int, Int, Int)]) -> [MLXArray] {
        let c = outDim
        let (pt, ph, pw) = (patchSize[0], patchSize[1], patchSize[2])
        var out: [MLXArray] = []
        for (i, grid) in gridSizes.enumerated() {
            let (f, h, w) = grid
            let seqLen = f * h * w
            var u = x[i, ..<seqLen]  // [L, outDim * pt * ph * pw]
            u = u.reshaped(f, h, w, pt, ph, pw, c)
            // [F', H', W', pt, ph, pw, C] -> [C, F'*pt, H'*ph, W'*pw]
            u = u.transposed(6, 0, 3, 1, 4, 2, 5)
            u = u.reshaped(c, f * pt, h * ph, w * pw)
            out.append(u)
        }
        return out
    }

    /// Precompute text embeddings (call once, reuse across steps).
    /// context: list of [L_text, textDim] -> [B, textLen, dim] in model dtype.
    public func embedText(_ context: [MLXArray]) -> MLXArray {
        let modelDtype = linearDtype(patchEmbeddingProj)
        var contextPadded: [MLXArray] = []
        for var ctx in context {
            let padLen = textLen - ctx.dim(0)
            if padLen > 0 {
                ctx = concatenated(
                    [ctx, MLXArray.zeros([padLen, ctx.dim(1)]).asType(ctx.dtype)],
                    axis: 0)
            }
            contextPadded.append(ctx)
        }
        let contextBatch = stacked(contextPadded)  // [B, textLen, textDim]
        let embedded = textEmbedding1(geluApproximate(textEmbedding0(contextBatch)))
        return embedded.asType(modelDtype)
    }

    /// Pre-compute cross-attention K/V for all blocks (one (k, v) per block).
    public func prepareCrossKV(_ context: MLXArray) -> [(MLXArray, MLXArray)] {
        blocks.map { $0.crossAttn.prepareKV(context) }
    }

    /// Pre-compute RoPE cos/sin for constant grid sizes.
    public func prepareRope(_ gridSizes: [(Int, Int, Int)]) -> (MLXArray, MLXArray) {
        let wDtype = linearDtype(patchEmbeddingProj)
        return ropePrecomputeCosSin(gridSizes: gridSizes, freqs: freqs, dtype: wDtype)
    }

    /// Forward pass.
    /// - xList: video latent tensors [C, F, H, W]
    /// - t: timestep tensor [B] (or scalar), or [B, L] for per-token timesteps
    /// - context: raw text features or pre-embedded tensor from `embedText`
    /// - seqLen: maximum sequence length for padding
    /// - y: optional I2V conditioning, channel-concatenated before patchify
    /// Returns denoised tensors [C, F, H, W] in float32.
    public func callAsFunction(
        _ xList: [MLXArray],
        t: MLXArray,
        context: WanTextContext,
        seqLen: Int,
        crossKVCaches: [(MLXArray, MLXArray)]? = nil,
        y: [MLXArray]? = nil,
        ropeCosSin: (MLXArray, MLXArray)? = nil
    ) -> [MLXArray] {
        // Detect identical inputs (CFG B=2) to avoid duplicate patchify work.
        // Check BEFORE I2V concat since concat creates new array objects.
        let batchSize = xList.count
        var allSame = batchSize > 1 && xList.dropFirst().allSatisfy { $0 === xList[0] }
        if allSame, let y {
            allSame = y.dropFirst().allSatisfy { $0 === y[0] }
        }

        // I2V: channel-concatenate conditioning y with noise x
        var xList = xList
        if let y {
            xList = zip(xList, y).map { concatenated([$0, $1], axis: 0) }
        }

        var x: MLXArray
        var gridSizes: [(Int, Int, Int)]
        var seqLensList: [Int]
        if allSame {
            // Patchify once and broadcast — saves a Linear projection per step
            var (p, gs) = patchify(xList[0])  // [1, L, dim]
            gridSizes = Array(repeating: gs, count: batchSize)
            seqLensList = Array(repeating: p.dim(1), count: batchSize)
            if p.dim(1) < seqLen {
                p = concatenated(
                    [p, MLXArray.zeros([1, seqLen - p.dim(1), dim]).asType(p.dtype)],
                    axis: 1)
            }
            x = broadcast(p, to: [batchSize, p.dim(1), p.dim(2)])
        } else {
            var patches: [MLXArray] = []
            gridSizes = []
            seqLensList = []
            for vid in xList {
                let (p, gs) = patchify(vid)
                patches.append(p)
                gridSizes.append(gs)
                seqLensList.append(p.dim(1))
            }
            x = concatenated(
                patches.map { p in
                    p.dim(1) < seqLen
                        ? concatenated(
                            [p, MLXArray.zeros([1, seqLen - p.dim(1), dim]).asType(p.dtype)],
                            axis: 1)
                        : p
                },
                axis: 0
            )  // [B, seqLen, dim]
        }

        // Time embedding: sinusoidal from precomputed inv_freq.
        var t = t
        if t.ndim == 0 {
            t = t.expandedDimensions(axis: 0)
        }

        let sinusoid = t.expandedDimensions(axis: -1).asType(.float32) * buffers.invFreq
        let sinEmb = concatenated([cos(sinusoid), sin(sinusoid)], axis: -1)

        let e: MLXArray
        let e0: MLXArray
        if t.ndim == 1 {
            // Standard T2V: scalar timestep per batch element [B]
            e = timeEmbedding1(silu(timeEmbedding0(sinEmb)))  // [B, dim]
            e0 = timeProjection(silu(e)).reshaped(batchSize, 1, 6, dim)
        } else {
            // I2V: per-token timesteps [B, L]
            e = timeEmbedding1(silu(timeEmbedding0(sinEmb)))  // [B, L, dim]
            e0 = timeProjection(silu(e)).reshaped(batchSize, -1, 6, dim)
        }

        // Text embedding: skip MLP if context is already embedded
        var contextBatch: MLXArray
        switch context {
        case .embedded(let embedded):
            contextBatch = embedded
            if contextBatch.dim(0) == 1 && batchSize > 1 {
                contextBatch = broadcast(
                    contextBatch,
                    to: [batchSize, contextBatch.dim(1), contextBatch.dim(2)])
            }
        case .raw(let raw):
            contextBatch = embedText(raw)
        }

        // Pre-compute attention mask from seqLens (constant across all blocks)
        var attnMask: MLXArray? = nil
        let wDtype = linearDtype(patchEmbeddingProj)
        if seqLensList.contains(where: { $0 < seqLen }) {
            attnMask = paddingMask(seqLens: seqLensList, total: seqLen, dtype: wDtype)
        }

        // Run transformer blocks
        for (i, block) in blocks.enumerated() {
            let kv = crossKVCaches?[i]
            x = block(
                x, e: e0, seqLens: seqLensList, gridSizes: gridSizes, freqs: freqs,
                context: contextBatch, contextLens: nil, crossKVCache: kv,
                ropeCosSin: ropeCosSin, attnMask: attnMask)
        }

        // Output head
        x = head(x, e)

        // Unpatchify
        return unpatchify(x, gridSizes: gridSizes).map { $0.asType(.float32) }
    }
}
