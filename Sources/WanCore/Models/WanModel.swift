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

/// Sequence-length threshold above which the Metal bf16 attention path is unstable
/// (nondeterministic NaNs / wrong values in the long-seq fused kernel + lazy-graph
/// buffer reuse — mlx-swift 0.31.4, Apple Silicon). Above it we switch on two
/// idiomatic mitigations: fp32 SDPA (in the attention modules) + per-block `eval`
/// (in `WanModel`). Below it (t2i, low-res, the A14B-validated paths) nothing
/// changes — the bf16 graph stays bit-identical. ~256² single-frame = seqLen 256
/// (safe); 512²×17f = 1280 (the first observed failure), so the gate sits between.
public let wanLargeSeq = 1024

/// EXPERIMENT KNOB (E15 / mlx-video diff): `wanLargeSeq` gates TWO mitigations — the fp32 SDPA
/// upcast (in the attention modules) AND the per-block `eval` (in `WanModel`/branches). The NaN
/// they mitigate was traced to the long-seq **fused-kernel dispatch race + lazy-graph buffer
/// reuse**, for which the per-block `eval` (graph chaining) is the actual fix — so the fp32
/// upcast may be REDUNDANT for correctness yet the memory/speed driver at large seqLen (fp32
/// drops out of the fused flash kernel that mlx-video rides in bf16). This flag disables ONLY the
/// upcast (per-block eval stays gated on `wanLargeSeq`), to run the bf16-fused path and discriminate
/// whether fp32 is load-bearing. `WAN_FP32_SDPA=0` → bf16 SDPA at all seqLens (mlx-video's behavior);
/// default (unset / non-"0") preserves the current fp32-upcast behavior. Decoupling matters: raising
/// `wanLargeSeq` alone would ALSO turn off per-block eval and un-bound the graph.
let wanForceFp32SdpaLargeSeq =
    ProcessInfo.processInfo.environment["WAN_FP32_SDPA"].map { $0 != "0" } ?? true

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
// `open` so adapter consumers IN OTHER PACKAGES (VACE Context Adapter, future ControlNet/T2I-Adapter)
// can subclass to add a parallel branch + use the embed/runBlocks(blockResiduals:)/finish seam, with
// the inherited backbone keys staying unprefixed (so the standard converted weights load unchanged).
open class WanModel: Module, @unchecked Sendable {
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
    /// Intermediate state threaded `embed` → `runBlocks` → `finish`. Exposing the phases lets a
    /// consumer (e.g. a VACE Context Adapter) run a parallel branch on the embedded latent `x` and
    /// the SAME per-block kwargs (`e0`/`seqLens`/`gridSizes`/`freqs`/`context`/`attnMask`), then
    /// inject per-block residuals — without re-implementing the forward. `callAsFunction` composes
    /// the three with no injection, so existing consumers are byte-for-byte unchanged.
    public struct ForwardState {
        public var x: MLXArray
        public let e: MLXArray
        public let e0: MLXArray
        public let seqLensList: [Int]
        public let gridSizes: [(Int, Int, Int)]
        public let contextBatch: MLXArray
        public let attnMask: MLXArray?
        public let batchSize: Int
    }

    public func callAsFunction(
        _ xList: [MLXArray],
        t: MLXArray,
        context: WanTextContext,
        seqLen: Int,
        crossKVCaches: [(MLXArray, MLXArray)]? = nil,
        y: [MLXArray]? = nil,
        ropeCosSin: (MLXArray, MLXArray)? = nil
    ) -> [MLXArray] {
        var s = embed(xList, t: t, context: context, seqLen: seqLen, y: y)
        runBlocks(&s, crossKVCaches: crossKVCaches, ropeCosSin: ropeCosSin)
        return finish(s)
    }

    /// Phase 1 — patchify + time/text embedding + attention mask. Returns the block inputs.
    public func embed(
        _ xList: [MLXArray],
        t: MLXArray,
        context: WanTextContext,
        seqLen: Int,
        y: [MLXArray]? = nil
    ) -> ForwardState {
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

        return ForwardState(
            x: x, e: e, e0: e0, seqLensList: seqLensList, gridSizes: gridSizes,
            contextBatch: contextBatch, attnMask: attnMask, batchSize: batchSize)
    }

    /// Phase 2 — run the transformer blocks. `blockResiduals[i]`, if present, is added to `x` AFTER
    /// block `i`: the generic ControlNet / T2I-Adapter / VACE injection seam (keys = main-block indices).
    public func runBlocks(
        _ s: inout ForwardState,
        crossKVCaches: [(MLXArray, MLXArray)]? = nil,
        ropeCosSin: (MLXArray, MLXArray)? = nil,
        blockResiduals: [Int: MLXArray]? = nil
    ) {
        // At large sequence length (multi-frame video on the GPU), bf16 attention on Metal is
        // numerically unstable — see `wanLargeSeq`: `eval` after each block BOUNDs the lazy graph
        // (mlx-lm-idiomatic graph-size control), resetting MLX's Metal buffer-reuse and suppressing
        // the nondeterministic long-seq NaN (paired with fp32 SDPA). Small-seq paths keep the
        // unbounded bf16 graph, so they stay bit-identical. (`WANCORE_DEBUG_NAN` adds a probe.)
        let evalEachBlock = s.x.dim(1) >= wanLargeSeq
        let debugNaN = ProcessInfo.processInfo.environment["WANCORE_DEBUG_NAN"] != nil
        for (i, block) in blocks.enumerated() {
            let kv = crossKVCaches?[i]
            s.x = block(
                s.x, e: s.e0, seqLens: s.seqLensList, gridSizes: s.gridSizes, freqs: freqs,
                context: s.contextBatch, contextLens: nil, crossKVCache: kv,
                ropeCosSin: ropeCosSin, attnMask: s.attnMask)
            if let r = blockResiduals?[i] { s.x = s.x + r }
            if evalEachBlock || debugNaN {
                eval(s.x)
                if debugNaN, !s.x.abs().max().item(Float.self).isFinite {
                    print("[WANCORE_DEBUG_NAN] first non-finite after block \(i)")
                    break
                }
            }
        }
    }

    /// Phase 3 — output head + unpatchify.
    public func finish(_ s: ForwardState) -> [MLXArray] {
        let x = head(s.x, s.e)
        return unpatchify(x, gridSizes: s.gridSizes).map { $0.asType(.float32) }
    }
}
