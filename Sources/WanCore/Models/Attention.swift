// 1:1 translation of mlx_video/models/wan_2/attention.py (mlx-video @ 87db56a).
// WanRMSNorm / WanLayerNorm / WanSelfAttention / WanCrossAttention.
// Dtype discipline carried verbatim: residual stream fp32, attention/FFN
// internals in the weight dtype, RoPE applied in fp32.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Compute dtype of a linear layer, handling QuantizedLinear.
public func linearDtype(_ layer: Linear) -> DType {
    if let quantized = layer as? QuantizedLinear {
        return quantized.scales.dtype
    }
    return layer.weight.dtype
}

/// RMS normalization with learnable scale.
class WanRMSNorm: Module {
    let eps: Float
    let weight: MLXArray

    init(_ dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self.weight = MLXArray.ones([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// LayerNorm computed in float32, with optional affine.
class WanLayerNorm: Module {
    let eps: Float
    let elementwiseAffine: Bool
    @ParameterInfo(key: "weight") var weight: MLXArray?
    @ParameterInfo(key: "bias") var bias: MLXArray?

    init(_ dim: Int, _ eps: Float = 1e-6, elementwiseAffine: Bool = false) {
        self.eps = eps
        self.elementwiseAffine = elementwiseAffine
        if elementwiseAffine {
            self._weight.wrappedValue = MLXArray.ones([dim])
            self._bias.wrappedValue = MLXArray.zeros([dim])
        } else {
            self._weight.wrappedValue = nil
            self._bias.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if elementwiseAffine {
            return MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
        } else {
            return MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps)
        }
    }
}

/// Self-attention with QK normalization and 3-way factorized RoPE.
class WanSelfAttention: Module {
    let dim: Int
    let numHeads: Int
    let headDim: Int
    let windowSize: (Int, Int)
    let scale: Float

    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "norm_q") var normQ: WanRMSNorm?
    @ModuleInfo(key: "norm_k") var normK: WanRMSNorm?

    init(
        _ dim: Int, _ numHeads: Int, windowSize: (Int, Int) = (-1, -1),
        qkNorm: Bool = true, eps: Float = 1e-6
    ) {
        precondition(dim % numHeads == 0)
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.windowSize = windowSize
        self.scale = pow(Float(headDim), -0.5)

        self._q.wrappedValue = Linear(dim, dim)
        self._k.wrappedValue = Linear(dim, dim)
        self._v.wrappedValue = Linear(dim, dim)
        self._o.wrappedValue = Linear(dim, dim)
        self._normQ.wrappedValue = qkNorm ? WanRMSNorm(dim, eps: eps) : nil
        self._normK.wrappedValue = qkNorm ? WanRMSNorm(dim, eps: eps) : nil
    }

    func callAsFunction(
        _ x: MLXArray,
        seqLens: [Int],
        gridSizes: [(Int, Int, Int)],
        freqs: MLXArray,
        ropeCosSin: (MLXArray, MLXArray)? = nil,
        attnMask: MLXArray? = nil
    ) -> MLXArray {
        let (b, s) = (x.dim(0), x.dim(1))
        let (n, d) = (numHeads, headDim)

        // Cast to compute dtype for efficient matmul (bf16, matching official autocast)
        let wDtype = linearDtype(q)
        let xW = x.asType(wDtype)

        var qProj = q(xW)
        var kProj = k(xW)
        if let normQ { qProj = normQ(qProj) }
        if let normK { kProj = normK(kProj) }

        qProj = qProj.reshaped(b, s, n, d)
        kProj = kProj.reshaped(b, s, n, d)
        let vProj = v(xW).reshaped(b, s, n, d)

        // RoPE in float32 for precision (official uses float64)
        qProj = ropeApply(
            qProj.asType(.float32), gridSizes: gridSizes, freqs: freqs,
            precomputedCosSin: ropeCosSin)
        kProj = ropeApply(
            kProj.asType(.float32), gridSizes: gridSizes, freqs: freqs,
            precomputedCosSin: ropeCosSin)

        // Cast back to weight dtype for efficient attention
        let qT = qProj.asType(wDtype).transposed(0, 2, 1, 3)
        let kT = kProj.asType(wDtype).transposed(0, 2, 1, 3)
        let vT = vProj.transposed(0, 2, 1, 3)

        // Use precomputed mask or build from seqLens
        var mask = attnMask
        if mask == nil && seqLens.contains(where: { $0 < s }) {
            mask = paddingMask(seqLens: seqLens, total: s, dtype: qT.dtype)
        }

        // Metal bf16 self-attention is unstable at large seqLen → fp32 SDPA there (the
        // QK^T/softmax-V chain; same house-style remedy WanVAE uses on the .cpu stream).
        // Small seqLen stays bf16, bit-identical. Paired with per-block eval in WanModel.
        let largeSeq = s >= wanLargeSeq
        let (qS, kS, vS) = largeSeq
            ? (qT.asType(.float32), kT.asType(.float32), vT.asType(.float32)) : (qT, kT, vT)
        let out = MLXFast.scaledDotProductAttention(
            queries: qS, keys: kS, values: vS, scale: scale,
            mask: largeSeq ? mask?.asType(.float32) : mask
        ).asType(wDtype)

        return o(out.transposed(0, 2, 1, 3).reshaped(b, s, -1))
    }
}

/// Cross-attention: Q from hidden states, K/V from text context.
class WanCrossAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "norm_q") var normQ: WanRMSNorm?
    @ModuleInfo(key: "norm_k") var normK: WanRMSNorm?

    init(_ dim: Int, _ numHeads: Int, qkNorm: Bool = true, eps: Float = 1e-6) {
        precondition(dim % numHeads == 0)
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)

        self._q.wrappedValue = Linear(dim, dim)
        self._k.wrappedValue = Linear(dim, dim)
        self._v.wrappedValue = Linear(dim, dim)
        self._o.wrappedValue = Linear(dim, dim)
        self._normQ.wrappedValue = qkNorm ? WanRMSNorm(dim, eps: eps) : nil
        self._normK.wrappedValue = qkNorm ? WanRMSNorm(dim, eps: eps) : nil
    }

    /// Pre-compute K and V projections for caching across denoising steps.
    /// Returns (k, v) each [B, N, L_ctx, D] ready for attention.
    func prepareKV(_ context: MLXArray) -> (MLXArray, MLXArray) {
        let b = context.dim(0)
        let (n, d) = (numHeads, headDim)
        let wDtype = linearDtype(k)
        let ctx = context.asType(wDtype)
        var kProj = k(ctx)
        if let normK { kProj = normK(kProj) }
        let kOut = kProj.reshaped(b, -1, n, d).transposed(0, 2, 1, 3)
        let vOut = v(ctx).reshaped(b, -1, n, d).transposed(0, 2, 1, 3)
        return (kOut, vOut)
    }

    func callAsFunction(
        _ x: MLXArray,
        context: MLXArray,
        contextLens: [Int]? = nil,
        kvCache: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        let b = x.dim(0)
        let (n, d) = (numHeads, headDim)

        let wDtype = linearDtype(q)
        var qProj = q(x.asType(wDtype))
        if let normQ { qProj = normQ(qProj) }
        let qT = qProj.reshaped(b, -1, n, d).transposed(0, 2, 1, 3)

        let kT: MLXArray
        let vT: MLXArray
        if let (kCached, vCached) = kvCache {
            kT = kCached
            vT = vCached
        } else {
            let ctx = context.asType(wDtype)
            var kProj = k(ctx)
            if let normK { kProj = normK(kProj) }
            kT = kProj.reshaped(b, -1, n, d).transposed(0, 2, 1, 3)
            vT = v(ctx).reshaped(b, -1, n, d).transposed(0, 2, 1, 3)
        }

        // Optional context masking
        var mask: MLXArray? = nil
        if let contextLens {
            mask = paddingMask(seqLens: contextLens, total: kT.dim(2), dtype: qT.dtype)
        }

        let out: MLXArray
        if let mask {
            out = MLXFast.scaledDotProductAttention(
                queries: qT, keys: kT, values: vT, scale: scale, mask: mask)
        } else {
            out = MLXFast.scaledDotProductAttention(
                queries: qT, keys: kT, values: vT, scale: scale, mask: nil)
        }

        return o(out.transposed(0, 2, 1, 3).reshaped(b, -1, n * d))
    }
}

/// Additive padding mask [B, 1, 1, total]: 0 for valid positions, -1e9 past
/// each sequence's length. (The Python builds this via in-place slice writes;
/// functional construction here, same values.)
func paddingMask(seqLens: [Int], total: Int, dtype: DType) -> MLXArray {
    let rows = seqLens.map { sl -> MLXArray in
        if sl < total {
            return concatenated([
                MLXArray.zeros([sl]),
                MLXArray.ones([total - sl]) * Float(-1e9),
            ])
        }
        return MLXArray.zeros([total])
    }
    return stacked(rows).reshaped(seqLens.count, 1, 1, total).asType(dtype)
}
