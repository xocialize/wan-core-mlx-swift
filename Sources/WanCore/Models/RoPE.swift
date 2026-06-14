// 1:1 translation of mlx_video/models/wan_2/rope.py (mlx-video @ 87db56a).
// 3-way factorized RoPE (temporal/height/width) over head_dim, stored as
// (cos, sin) pairs. Frequency tables are computed in Float64 (numpy parity)
// before the float32 cast.

import Foundation
import MLX

/// Precompute RoPE frequency parameters as (cos, sin) pairs.
/// Returns shape [maxSeqLen, dim / 2, 2].
public func ropeParams(_ maxSeqLen: Int, _ dim: Int, theta: Double = 10000.0) -> MLXArray {
    precondition(dim % 2 == 0)
    let halfD = dim / 2
    // float64 throughout, cast to float32 at the end (matches numpy reference)
    var flat = [Float](repeating: 0, count: maxSeqLen * halfD * 2)
    for pos in 0..<maxSeqLen {
        for j in 0..<halfD {
            let freq = Double(pos) / pow(theta, Double(2 * j) / Double(dim))
            flat[(pos * halfD + j) * 2 + 0] = Float(cos(freq))
            flat[(pos * halfD + j) * 2 + 1] = Float(sin(freq))
        }
    }
    return MLXArray(flat, [maxSeqLen, halfD, 2])
}

/// Apply 3-way factorized RoPE to a Q or K tensor of shape [B, L, numHeads, headDim].
func ropeApply(
    _ x: MLXArray,
    gridSizes: [(Int, Int, Int)],
    freqs: MLXArray,
    precomputedCosSin: (MLXArray, MLXArray)? = nil
) -> MLXArray {
    let (b, s, n, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let halfD = d / 2

    if let (cosF, sinF) = precomputedCosSin {
        let (f0, h0, w0) = gridSizes[0]
        let seqLen = f0 * h0 * w0
        let allSameGrid = b <= 1 || gridSizes.allSatisfy { $0 == gridSizes[0] }

        if allSameGrid {
            // Vectorized path: apply RoPE to all batch elements at once
            let xSeq = x[0..., ..<seqLen].reshaped(b, seqLen, n, halfD, 2)
            let xReal = xSeq[.ellipsis, 0]
            let xImag = xSeq[.ellipsis, 1]
            let outReal = xReal * cosF - xImag * sinF
            let outImag = xReal * sinF + xImag * cosF
            var xRotated = stacked([outReal, outImag], axis: -1).reshaped(b, seqLen, n, d)
            if seqLen < s {
                xRotated = concatenated([xRotated, x[0..., seqLen...]], axis: 1)
            }
            return xRotated
        } else {
            // Per-element path for mixed grid sizes
            var outputs: [MLXArray] = []
            for i in 0..<b {
                let (f, h, w) = gridSizes[i]
                let sl = f * h * w
                let xI = x[i, ..<sl].reshaped(sl, n, halfD, 2)
                let xReal = xI[.ellipsis, 0]
                let xImag = xI[.ellipsis, 1]
                let outReal = xReal * cosF - xImag * sinF
                let outImag = xReal * sinF + xImag * cosF
                var xRotated = stacked([outReal, outImag], axis: -1).reshaped(sl, n, d)
                if sl < s {
                    xRotated = concatenated([xRotated, x[i, sl...]], axis: 0)
                }
                outputs.append(xRotated)
            }
            return stacked(outputs)
        }
    }

    // Cast freqs to input dtype to prevent float32 promotion cascade
    var freqs = freqs
    if freqs.dtype != x.dtype {
        freqs = freqs.asType(x.dtype)
    }

    // Split frequency dimensions: temporal gets more capacity
    let dT = halfD - 2 * (halfD / 3)
    let dH = halfD / 3
    let dW = halfD / 3

    let freqsT = freqs[0..., ..<dT]
    let freqsH = freqs[0..., dT..<(dT + dH)]
    let freqsW = freqs[0..., (dT + dH)..<(dT + dH + dW)]

    var outputs: [MLXArray] = []
    for i in 0..<b {
        let (f, h, w) = gridSizes[i]
        let seqLen = f * h * w

        let xI = x[i, ..<seqLen].reshaped(seqLen, n, halfD, 2)

        // Build per-position frequencies by expanding along grid dims
        let ft = broadcast(freqsT[..<f].reshaped(f, 1, 1, dT, 2), to: [f, h, w, dT, 2])
        let fh = broadcast(freqsH[..<h].reshaped(1, h, 1, dH, 2), to: [f, h, w, dH, 2])
        let fw = broadcast(freqsW[..<w].reshaped(1, 1, w, dW, 2), to: [f, h, w, dW, 2])

        let freqsI = concatenated([ft, fh, fw], axis: 3).reshaped(seqLen, 1, halfD, 2)

        // (a + bi) * (cos + sin*i) = (a*cos - b*sin) + (a*sin + b*cos)i
        let cosF = freqsI[.ellipsis, 0]
        let sinF = freqsI[.ellipsis, 1]

        let xReal = xI[.ellipsis, 0]
        let xImag = xI[.ellipsis, 1]

        let outReal = xReal * cosF - xImag * sinF
        let outImag = xReal * sinF + xImag * cosF

        var xRotated = stacked([outReal, outImag], axis: -1).reshaped(seqLen, n, d)

        if seqLen < s {
            xRotated = concatenated([xRotated, x[i, seqLen...]], axis: 0)
        }
        outputs.append(xRotated)
    }
    return stacked(outputs)
}

/// Precompute cos/sin frequency tensors for constant grid sizes.
/// Returns (cosF, sinF), each [seqLen, 1, halfD].
public func ropePrecomputeCosSin(
    gridSizes: [(Int, Int, Int)], freqs: MLXArray, dtype: DType = .float32
) -> (MLXArray, MLXArray) {
    var freqs = freqs
    if freqs.dtype != dtype {
        freqs = freqs.asType(dtype)
    }

    let (f, h, w) = gridSizes[0]
    let seqLen = f * h * w
    let halfD = freqs.dim(1)

    let dT = halfD - 2 * (halfD / 3)
    let dH = halfD / 3
    let dW = halfD / 3

    let freqsT = freqs[0..., ..<dT]
    let freqsH = freqs[0..., dT..<(dT + dH)]
    let freqsW = freqs[0..., (dT + dH)..<(dT + dH + dW)]

    let ft = broadcast(freqsT[..<f].reshaped(f, 1, 1, dT, 2), to: [f, h, w, dT, 2])
    let fh = broadcast(freqsH[..<h].reshaped(1, h, 1, dH, 2), to: [f, h, w, dH, 2])
    let fw = broadcast(freqsW[..<w].reshaped(1, 1, w, dW, 2), to: [f, h, w, dW, 2])

    let freqsI = concatenated([ft, fh, fw], axis: 3).reshaped(seqLen, 1, halfD, 2)
    return (freqsI[.ellipsis, 0], freqsI[.ellipsis, 1])
}
