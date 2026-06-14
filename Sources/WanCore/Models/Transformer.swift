// 1:1 translation of mlx_video/models/wan_2/transformer.py (mlx-video @ 87db56a).
// WanAttentionBlock + WanFFN. Modulation math stays float32 so the residual
// stream is promoted to float32 across all layers (matching the reference's
// autocast(float32) discipline).

import Foundation
import MLX
import MLXNN

/// Wan transformer block with learned modulation, self-attn, cross-attn, and FFN.
public class WanAttentionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: WanLayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: WanSelfAttention
    @ModuleInfo(key: "norm3") var norm3: WanLayerNorm?
    @ModuleInfo(key: "cross_attn") var crossAttn: WanCrossAttention
    @ModuleInfo(key: "norm2") var norm2: WanLayerNorm
    @ModuleInfo(key: "ffn") var ffn: WanFFN
    /// Learned modulation: 6 vectors for scale/shift/gate (kept in float32).
    let modulation: MLXArray

    init(
        dim: Int,
        ffnDim: Int,
        numHeads: Int,
        windowSize: (Int, Int) = (-1, -1),
        qkNorm: Bool = true,
        crossAttnNorm: Bool = false,
        eps: Float = 1e-6
    ) {
        self._norm1.wrappedValue = WanLayerNorm(dim, eps)
        self._selfAttn.wrappedValue = WanSelfAttention(
            dim, numHeads, windowSize: windowSize, qkNorm: qkNorm, eps: eps)
        self._norm3.wrappedValue =
            crossAttnNorm ? WanLayerNorm(dim, eps, elementwiseAffine: true) : nil
        self._crossAttn.wrappedValue = WanCrossAttention(dim, numHeads, qkNorm: qkNorm, eps: eps)
        self._norm2.wrappedValue = WanLayerNorm(dim, eps)
        self._ffn.wrappedValue = WanFFN(dim: dim, ffnDim: ffnDim)
        self.modulation = (MLXRandom.normal([1, 6, dim]) * pow(Float(dim), -0.5))
            .asType(.float32)
    }

    public func callAsFunction(
        _ x: MLXArray,
        e: MLXArray,
        seqLens: [Int],
        gridSizes: [(Int, Int, Int)],
        freqs: MLXArray,
        context: MLXArray,
        contextLens: [Int]? = nil,
        crossKVCache: (MLXArray, MLXArray)? = nil,
        ropeCosSin: (MLXArray, MLXArray)? = nil,
        attnMask: MLXArray? = nil
    ) -> MLXArray {
        var x = x
        // Modulation in float32; type promotion keeps the residual stream
        // float32 throughout (gate * output + x -> float32).
        let mod = modulation + e  // [B, L_e, 6, dim]
        let e0 = mod[0..., 0..., 0, 0...]  // shift for self-attn
        let e1 = mod[0..., 0..., 1, 0...]  // scale for self-attn
        let e2 = mod[0..., 0..., 2, 0...]  // gate for self-attn
        let e3 = mod[0..., 0..., 3, 0...]  // shift for ffn
        let e4 = mod[0..., 0..., 4, 0...]  // scale for ffn
        let e5 = mod[0..., 0..., 5, 0...]  // gate for ffn

        // Self-attention with modulation (hidden state stays in w_dtype)
        var xMod = norm1(x) * (1 + e1) + e0
        var y = selfAttn(
            xMod, seqLens: seqLens, gridSizes: gridSizes, freqs: freqs,
            ropeCosSin: ropeCosSin, attnMask: attnMask)
        x = x + y * e2

        // Cross-attention (no modulation, just norm)
        let xCross = norm3.map { $0(x) } ?? x
        x = x + crossAttn(xCross, context: context, contextLens: contextLens, kvCache: crossKVCache)

        // FFN with modulation
        xMod = norm2(x) * (1 + e4) + e3
        y = ffn(xMod)
        x = x + y * e5

        return x
    }
}

/// Gated feed-forward network with GELU(tanh) activation.
class WanFFN: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(dim: Int, ffnDim: Int) {
        self._fc1.wrappedValue = Linear(dim, ffnDim)
        self._fc2.wrappedValue = Linear(ffnDim, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Cast to compute dtype for efficient matmul (bf16 matching official autocast)
        let xW = x.asType(linearDtype(fc1))
        return fc2(geluApproximate(fc1(xW)))
    }
}
