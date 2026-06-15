// Wan2.2 VAE (z_dim=48, compression 4×16×16) — the high-compression sibling of
// the 16-ch `WanVAE`. 1:1 port of mlx-video `wan_2/vae22.py` (@87db56a), reusing
// this module's shared primitives (CausalConv3d / RMS_norm / ResidualBlock /
// AttentionBlock / Resample / FeatCache+Rep / StreamingDecode) — the wan-core
// payoff. Net-new here: DupUp3D / AvgDown3D shortcuts, the Up/Down stage
// wrappers, Decoder3d/Encoder3d assembly, 2×2 patchify, and the 48-vec stats.
//
// Layout: channels-last [B, T, H, W, C] throughout (matches vae22.py / WanVAE).
// The VAE runs in float32 (official Wan2.2; utils.load_vae_decoder upcasts).
// Temporal upsample uses the SAME first-chunk frame-0 skip as the 16-ch path
// (E11): vae22.py IS the reference that fix was ported from.

import Foundation
import MLX
import MLXNN

// MARK: - Per-channel latent normalization (z_dim=48)

/// vae22.py VAE22_MEAN — 48-vector per-channel latent mean (the normalization
/// landmine; applied at exactly one site, like the 16-ch VAE_MEAN).
public let VAE22_MEAN: [Float] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]

/// vae22.py VAE22_STD — 48-vector per-channel latent std.
public let VAE22_STD: [Float] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]

/// Denormalize a normalized 48-ch latent for decode: `z * std + mean`
/// (channels-last, broadcast over the trailing C axis). Mirrors
/// vae22.py `denormalize_latents`.
public func denormalizeLatents22(_ z: MLXArray) -> MLXArray {
    let mean = MLXArray(VAE22_MEAN)  // [48]
    let std = MLXArray(VAE22_STD)
    return z * std + mean
}

/// Normalize an encoded 48-ch latent: `(z - mean) / std`. Mirrors
/// vae22.py `normalize_latents`.
public func normalizeLatents22(_ z: MLXArray) -> MLXArray {
    let mean = MLXArray(VAE22_MEAN)
    let std = MLXArray(VAE22_STD)
    return (z - mean) / std
}

// MARK: - DupUp3D (param-free upsample shortcut)

/// Upsample by duplicating channels + reshaping (no learnable params) — the
/// vae22 decoder's residual shortcut. Channels-last [B,T,H,W,C]. 1:1 port of
/// vae22.py `DupUp3D`.
public final class DupUp3D: Module, @unchecked Sendable {
    public let inChannels: Int
    public let outChannels: Int
    public let factorT: Int
    public let factorS: Int
    public let repeats: Int

    public init(_ inChannels: Int, _ outChannels: Int, factorT: Int, factorS: Int = 1) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.factorT = factorT
        self.factorS = factorS
        let factor = factorT * factorS * factorS
        self.repeats = outChannels * factor / inChannels
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, firstChunk: Bool = false) -> MLXArray {
        let (b, t, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        // Repeat channels (np.repeat semantics): [B,T,H,W,C*repeats]
        var y = repeated(x, count: repeats, axis: -1)
        // → [B,T,H,W, outC, factorT, factorS, factorS]
        y = y.reshaped(b, t, h, w, outChannels, factorT, factorS, factorS)
        // interleave → [B, T, factorT, H, factorS, W, factorS, outC]
        y = y.transposed(0, 1, 5, 2, 6, 3, 7, 4)
        // → [B, T*factorT, H*factorS, W*factorS, outC]
        y = y.reshaped(b, t * factorT, h * factorS, w * factorS, outChannels)
        if firstChunk {
            // drop the (factorT-1) extra leading temporal frames on the first chunk
            y = y[0..., (factorT - 1)...]
        }
        return y
    }
}

// MARK: - AvgDown3D (param-free downsample shortcut — encoder)

/// Downsample by grouping spatial/temporal factors and averaging (inverse of
/// DupUp3D, no params). 1:1 port of vae22.py `AvgDown3D`.
public final class AvgDown3D: Module, @unchecked Sendable {
    public let inChannels: Int
    public let outChannels: Int
    public let factorT: Int
    public let factorS: Int
    public let groupSize: Int

    public init(_ inChannels: Int, _ outChannels: Int, factorT: Int, factorS: Int = 1) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.factorT = factorT
        self.factorS = factorS
        let factor = factorT * factorS * factorS
        self.groupSize = inChannels * factor / outChannels
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        let b = x.dim(0)
        var t = x.dim(1)
        let (h, w, c) = (x.dim(2), x.dim(3), x.dim(4))
        // Pad temporal (front) up to a multiple of factorT.
        let padT = (factorT - t % factorT) % factorT
        if padT > 0 {
            x = padded(x, widths: [.init((0, 0)), .init((padT, 0)), .init((0, 0)),
                                   .init((0, 0)), .init((0, 0))])
            t += padT
        }
        let ft = factorT, fs = factorS
        x = x.reshaped(b, t / ft, ft, h / fs, fs, w / fs, fs, c)
        // → [B, T', H', W', C, ft, fs, fs]
        x = x.transposed(0, 1, 3, 5, 7, 2, 4, 6)
        x = x.reshaped(b, t / ft, h / fs, w / fs, c * ft * fs * fs)
        x = x.reshaped(b, t / ft, h / fs, w / fs, outChannels, groupSize)
        return x.mean(axis: -1)
    }
}

// MARK: - V22CausalConv3d (channels-LAST)

/// 3D causal convolution, channels-last [B,T,H,W,C]. 1:1 port of vae22.py
/// `CausalConv3d` — decomposes the 3D conv into per-frame 2D convs (the oracle's
/// memory pattern). NOTE: vae22's causal temporal pad is `2*padding[0]` (NOT the
/// 16-ch `k-stride`), so this is a distinct class from the 16-ch `CausalConv3d`.
/// Weight `[O, kd, kh, kw, I]`.
public final class V22CausalConv3d: Module, @unchecked Sendable {
    public let kernelSize: (Int, Int, Int)
    public let stride: (Int, Int, Int)
    let causalPadT: Int
    let padH: Int
    let padW: Int
    public let weight: MLXArray
    public let bias: MLXArray

    public init(
        _ inChannels: Int, _ outChannels: Int, _ kernelSize: (Int, Int, Int),
        stride: (Int, Int, Int) = (1, 1, 1), padding: (Int, Int, Int) = (0, 0, 0)
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.causalPadT = 2 * padding.0
        self.padH = padding.1
        self.padW = padding.2
        self.weight = MLXArray.zeros(
            [outChannels, kernelSize.0, kernelSize.1, kernelSize.2, inChannels])
        self.bias = MLXArray.zeros([outChannels])
        super.init()
    }

    /// Scalar kernel/padding spelling (vae22's `CausalConv3d(in, out, 3, padding=1)`).
    public convenience init(_ inC: Int, _ outC: Int, _ k: Int, stride: Int = 1, padding: Int = 0) {
        self.init(inC, outC, (k, k, k), stride: (stride, stride, stride),
                  padding: (padding, padding, padding))
    }

    /// x: [B, T, H, W, C]. `cacheX` = prev-chunk trailing frames (chunked/streaming).
    public func callAsFunction(_ input: MLXArray, cacheX: MLXArray? = nil) -> MLXArray {
        var x = input
        let b = x.dim(0)
        let c = x.dim(4)
        let (kd, kh, kw) = kernelSize

        // 1×1×1 fast path: pointwise conv over the channel axis, per frame.
        if kd == 1 && kh == 1 && kw == 1 {
            let t = x.dim(1)
            let (h, w) = (x.dim(2), x.dim(3))
            let xFlat = x.reshaped(b * t, h, w, c)
            let w2d = weight[0..., 0, 0..., 0..., 0...]  // [O, 1, 1, I]
            let y = conv2d(xFlat, w2d) + bias
            return y.reshaped(b, t, y.dim(1), y.dim(2), -1)
        }

        // Causal temporal pad (prepend cached frames, then zero-pad the remainder).
        var padNeeded = causalPadT
        if let cacheX, padNeeded > 0 {
            x = concatenated([cacheX, x], axis: 1)
            padNeeded -= cacheX.dim(1)
        }
        if padNeeded > 0 {
            let padT = MLXArray.zeros([b, padNeeded, x.dim(2), x.dim(3), c], dtype: x.dtype)
            x = concatenated([padT, x], axis: 1)
        }
        // Spatial pad.
        if padH > 0 || padW > 0 {
            x = padded(x, widths: [.init((0, 0)), .init((0, 0)), .init((padH, padH)),
                                   .init((padW, padW)), .init((0, 0))])
        }

        let tPadded = x.dim(1)
        let tOut = (tPadded - kd) / stride.0 + 1
        // Decompose 3D conv into a sum of per-temporal-position 2D convs.
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(tOut)
        for t in 0..<tOut {
            let tStart = t * stride.0
            var accum: MLXArray? = nil
            for d in 0..<kd {
                let frame = x[0..., tStart + d]        // [B, Hp, Wp, C]
                let w2d = weight[0..., d, 0..., 0..., 0...]  // [O, kh, kw, I]
                let convOut = conv2d(frame, w2d, stride: .init((stride.1, stride.2)))
                accum = accum == nil ? convOut : accum! + convOut
            }
            outputs.append(accum! + bias)
        }
        return stacked(outputs, axis: 1)  // [B, T_out, H_out, W_out, O]
    }
}

// MARK: - V22RMSNorm (channels-LAST)

/// vae22.py `RMS_norm` — actually F.normalize (L2 over the channel axis) × scale ×
/// gamma, channels-last. Distinct from the 16-ch `RMS_norm` (layout/idiom).
public final class V22RMSNorm: Module, @unchecked Sendable {
    public let scale: Float
    public let gamma: MLXArray

    public init(_ dim: Int) {
        self.scale = Float(Double(dim).squareRoot())
        self.gamma = MLXArray.ones([dim])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let l2sq = (x * x).sum(axis: -1, keepDims: true)
        return x * rsqrt(maximum(l2sq, MLXArray(Float(1e-24)))) * scale * gamma
    }
}
