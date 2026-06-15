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
import MLXFast
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

// MARK: - V22ResidualBlock (channels-LAST)

/// The Sequential layers inside a vae22 ResidualBlock (channels-last). PyTorch
/// nn.Sequential indices 0/2/3/6 = RMS_norm / CausalConv3d / RMS_norm / CausalConv3d
/// (1,4 = SiLU, 5 = Dropout — no params). Named `layer_N` to match the checkpoint
/// keys. Reuses the shared `CACHE_T` / `FeatCache` / `FeatIdx` (layout-agnostic).
public final class V22ResidualBlockLayers: Module, @unchecked Sendable {
    @ModuleInfo(key: "layer_0") public var layer0: V22RMSNorm
    @ModuleInfo(key: "layer_2") public var layer2: V22CausalConv3d
    @ModuleInfo(key: "layer_3") public var layer3: V22RMSNorm
    @ModuleInfo(key: "layer_6") public var layer6: V22CausalConv3d

    public init(_ inDim: Int, _ outDim: Int) {
        self._layer0.wrappedValue = V22RMSNorm(inDim)
        self._layer2.wrappedValue = V22CausalConv3d(inDim, outDim, 3, padding: 1)
        self._layer3.wrappedValue = V22RMSNorm(outDim)
        self._layer6.wrappedValue = V22CausalConv3d(outDim, outDim, 3, padding: 1)
        super.init()
    }

    /// CausalConv3d with temporal feature-caching for chunked decode (axis 1 = T).
    private func convCached(_ conv: V22CausalConv3d, _ x: MLXArray, _ fc: FeatCache, _ fi: FeatIdx)
        -> MLXArray
    {
        let idx = fi.value
        var cacheX = x[0..., Swift.max(0, x.dim(1) - CACHE_T)...]
        if cacheX.dim(1) < 2, let cached = fc[idx] {
            cacheX = concatenated([cached[0..., (cached.dim(1) - 1)...], cacheX], axis: 1)
        }
        let out = conv(x, cacheX: fc[idx])  // fc[idx]=nil first chunk → zero-pad
        fc[idx] = cacheX
        fi.value += 1
        return out
    }

    public func callAsFunction(
        _ x0: MLXArray, featCache: FeatCache? = nil, featIdx: FeatIdx? = nil
    ) -> MLXArray {
        var x = layer0(x0)
        x = silu(x)
        if let fc = featCache, let fi = featIdx { x = convCached(layer2, x, fc, fi) }
        else { x = layer2(x) }
        eval(x)  // eval between convs to bound graph size (vae22.py mx.eval)
        x = layer3(x)
        x = silu(x)
        if let fc = featCache, let fi = featIdx { x = convCached(layer6, x, fc, fi) }
        else { x = layer6(x) }
        return x
    }
}

/// vae22 ResidualBlock: `residual` path + a 1×1 `shortcut` conv when dims change.
public final class V22ResidualBlock: Module, @unchecked Sendable {
    @ModuleInfo(key: "residual") public var residual: V22ResidualBlockLayers
    @ModuleInfo(key: "shortcut") public var shortcut: V22CausalConv3d?

    public init(_ inDim: Int, _ outDim: Int) {
        self._residual.wrappedValue = V22ResidualBlockLayers(inDim, outDim)
        self._shortcut.wrappedValue = inDim != outDim ? V22CausalConv3d(inDim, outDim, 1) : nil
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, featCache: FeatCache? = nil, featIdx: FeatIdx? = nil
    ) -> MLXArray {
        let h = shortcut.map { $0(x) } ?? x
        return residual(x, featCache: featCache, featIdx: featIdx) + h
    }
}

// MARK: - V22AttentionBlock (channels-LAST, 2D self-attn per frame)

/// vae22.py `AttentionBlock` — single-head 2D self-attention applied per frame.
/// QKV/proj are raw 1×1 conv params (`to_qkv_weight`/`proj_weight`, NOT Conv2d
/// submodules) to match the checkpoint keys. SDPA forced to the `.cpu` stream for
/// strict fp32 (donor lesson L10 — Metal loses ~3-4 bits on the QK^T/softmax-V chain).
public final class V22AttentionBlock: Module, @unchecked Sendable {
    public let dim: Int
    @ModuleInfo(key: "norm") public var norm: V22RMSNorm
    @ParameterInfo(key: "to_qkv_weight") public var toQkvWeight: MLXArray
    @ParameterInfo(key: "to_qkv_bias") public var toQkvBias: MLXArray
    @ParameterInfo(key: "proj_weight") public var projWeight: MLXArray
    @ParameterInfo(key: "proj_bias") public var projBias: MLXArray

    public init(_ dim: Int) {
        self.dim = dim
        self._norm.wrappedValue = V22RMSNorm(dim)
        self._toQkvWeight.wrappedValue = MLXArray.zeros([3 * dim, 1, 1, dim])
        self._toQkvBias.wrappedValue = MLXArray.zeros([3 * dim])
        self._projWeight.wrappedValue = MLXArray.zeros([dim, 1, 1, dim])
        self._projBias.wrappedValue = MLXArray.zeros([dim])
        super.init()
    }

    /// x: [B, T, H, W, C].
    public func callAsFunction(_ x0: MLXArray) -> MLXArray {
        let (b, t, h, w, c) = (x0.dim(0), x0.dim(1), x0.dim(2), x0.dim(3), x0.dim(4))
        let identity = x0
        var x = x0.reshaped(b * t, h, w, c)
        x = norm(x)
        // 1×1 conv = linear over channels: [BT,H,W,C] → [BT,H,W,3C].
        let qkv = (conv2d(x, toQkvWeight) + toQkvBias).reshaped(b * t, h * w, 3 * c)
        let parts = split(qkv, parts: 3, axis: -1)  // each [BT, HW, C]
        let q = parts[0][0..., .newAxis, 0..., 0...]  // [BT, 1, HW, C]
        let k = parts[1][0..., .newAxis, 0..., 0...]
        let v = parts[2][0..., .newAxis, 0..., 0...]
        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: Float(pow(Double(c), -0.5)), mask: nil, stream: .cpu)
        out = out.squeezed(axis: 1).reshaped(b * t, h, w, c)
        out = conv2d(out, projWeight) + projBias
        return out.reshaped(b, t, h, w, c) + identity
    }
}

// MARK: - V22Resample (channels-LAST spatial ± temporal up/downsample)

/// vae22.py `Resample` — nearest-neighbor 2× spatial up/down with optional temporal
/// up/down via a `time_conv` CausalConv3d. `resample_weight`/`resample_bias` are raw
/// 3×3 Conv2d params. Spatial pass is chunked (8 frames) with an `eval` per chunk to
/// bound graph size (the watchdog lever). The E11 first-chunk temporal-upsample skip
/// lives in the `upsample3d` branch.
public final class V22Resample: Module, @unchecked Sendable {
    public let dim: Int
    public let mode: String
    @ParameterInfo(key: "resample_weight") public var resampleWeight: MLXArray
    @ParameterInfo(key: "resample_bias") public var resampleBias: MLXArray
    @ModuleInfo(key: "time_conv") public var timeConv: V22CausalConv3d?

    public init(_ dim: Int, mode: String) {
        self.dim = dim
        self.mode = mode
        self._resampleWeight.wrappedValue = MLXArray.zeros([dim, 3, 3, dim])
        self._resampleBias.wrappedValue = MLXArray.zeros([dim])
        switch mode {
        case "upsample3d":
            self._timeConv.wrappedValue = V22CausalConv3d(dim, dim * 2, (3, 1, 1), padding: (1, 0, 0))
        case "downsample3d":
            self._timeConv.wrappedValue = V22CausalConv3d(
                dim, dim, (3, 1, 1), stride: (2, 1, 1), padding: (0, 0, 0))
        default:
            self._timeConv.wrappedValue = nil
        }
        super.init()
    }

    /// Nearest-neighbor 2× spatial upsample. x: [N, H, W, C].
    private func upsample2x(_ x: MLXArray) -> MLXArray {
        repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
    }
    /// 3×3 Conv2d with symmetric padding=1. x: [N, H, W, C].
    private func conv2dPad1(_ x: MLXArray) -> MLXArray {
        let xp = padded(x, widths: [.init((0, 0)), .init((1, 1)), .init((1, 1)), .init((0, 0))])
        return conv2d(xp, resampleWeight) + resampleBias
    }
    /// Strided 3×3 Conv2d for downsampling — ZeroPad2d((0,1,0,1)) then stride 2.
    private func downsampleConv2d(_ x: MLXArray) -> MLXArray {
        let xp = padded(x, widths: [.init((0, 0)), .init((0, 1)), .init((0, 1)), .init((0, 0))])
        return conv2d(xp, resampleWeight, stride: .init((2, 2))) + resampleBias
    }

    /// x: [B, T, H, W, C].
    public func callAsFunction(
        _ input: MLXArray, firstChunk: Bool = false,
        featCache: FeatCache? = nil, featIdx: FeatIdx? = nil
    ) -> MLXArray {
        var x = input
        let b = x.dim(0)
        var t = x.dim(1)
        let (h, w, c) = (x.dim(2), x.dim(3), x.dim(4))

        // --- Temporal upsample (before spatial) ---
        if mode == "upsample3d", let tc = timeConv {
            if firstChunk && t > 1 {
                let firstFrame = x[0..., 0..<1]
                let tcOut = tc(x[0..., 1...]).reshaped(b, t - 1, h, w, 2, c)
                let stream0 = tcOut[0..., 0..., 0..., 0..., 0, 0...]
                let stream1 = tcOut[0..., 0..., 0..., 0..., 1, 0...]
                let interleaved = stacked([stream0, stream1], axis: 2)
                    .reshaped(b, (t - 1) * 2, h, w, c)
                x = concatenated([firstFrame, interleaved], axis: 1)
            } else {
                let tcOut = tc(x).reshaped(b, t, h, w, 2, c)
                let stream0 = tcOut[0..., 0..., 0..., 0..., 0, 0...]
                let stream1 = tcOut[0..., 0..., 0..., 0..., 1, 0...]
                x = stacked([stream0, stream1], axis: 2).reshaped(b, t * 2, h, w, c)
            }
            eval(x)
            t = x.dim(1)
        }

        // --- Spatial up/down (chunked for upsample) ---
        if mode == "upsample2d" || mode == "upsample3d" {
            var chunks: [MLXArray] = []
            var tStart = 0
            while tStart < t {
                let tEnd = Swift.min(tStart + 8, t)
                var xc = x[0..., tStart..<tEnd].reshaped(-1, h, w, c)
                xc = conv2dPad1(upsample2x(xc))
                eval(xc)
                chunks.append(xc)
                tStart += 8
            }
            x = concatenated(chunks, axis: 0)
            x = x.reshaped(b, t, x.dim(1), x.dim(2), c)
        } else if mode == "downsample2d" || mode == "downsample3d" {
            let xf = downsampleConv2d(x.reshaped(b * t, h, w, c))
            eval(xf)
            x = xf.reshaped(b, t, xf.dim(1), xf.dim(2), c)
        }

        // --- Temporal downsample (after spatial) ---
        if mode == "downsample3d", let tc = timeConv {
            if let fc = featCache, let fi = featIdx {
                let idx = fi.value
                if let cached = fc[idx] {
                    let saveX = x[0..., (x.dim(1) - 1)...]
                    x = tc(concatenated([cached[0..., (cached.dim(1) - 1)...], x], axis: 1))
                    fc[idx] = saveX
                } else {
                    fc[idx] = x  // first chunk: store, skip time_conv
                }
                fi.value += 1
            } else if t > 1 {
                x = tc(x)
            }
            eval(x)
        }
        return x
    }
}

// MARK: - unpatchify (2×2 spatial depth-to-space, channels-LAST)

/// vae22.py `_unpatchify` — [B,T,H,W,C·p·p] → [B,T,H·p,W·p,C]. The decoder head
/// emits 12 packed channels; unpatchify spreads them to 3 RGB at 2× spatial.
public func unpatchify22(_ x: MLXArray, patchSize: Int = 2) -> MLXArray {
    if patchSize == 1 { return x }
    let (b, t, h, w, cPacked) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
    let c = cPacked / (patchSize * patchSize)
    var y = x.reshaped(b, t, h, w, c, patchSize, patchSize)  // unpack (C, r, q)
    y = y.transposed(0, 1, 2, 6, 3, 5, 4)  // [B, T, H, q, W, r, C]
    return y.reshaped(b, t, h * patchSize, w * patchSize, c)
}

// MARK: - V22UpResidualBlock (decoder upsampling stage)

/// vae22.py `Up_ResidualBlock` — (num_res_blocks) ResidualBlocks + optional Resample,
/// plus a param-free DupUp3D `avg_shortcut` when this stage upsamples. The `upsamples`
/// list keys (`upsamples.0`, …) match the checkpoint.
public final class V22UpResidualBlock: Module, @unchecked Sendable {
    public let upFlag: Bool
    @ModuleInfo(key: "avg_shortcut") public var avgShortcut: DupUp3D?
    public let upsamples: [Module]

    public init(
        inDim: Int, outDim: Int, numResBlocks: Int,
        temperalUpsample: Bool = false, upFlag: Bool = false
    ) {
        self.upFlag = upFlag
        var blocks = [Module]()
        var dimIn = inDim
        for _ in 0..<numResBlocks {
            blocks.append(V22ResidualBlock(dimIn, outDim))
            dimIn = outDim
        }
        if upFlag {
            blocks.append(V22Resample(outDim, mode: temperalUpsample ? "upsample3d" : "upsample2d"))
        }
        self.upsamples = blocks
        super.init()
        self._avgShortcut.wrappedValue =
            upFlag ? DupUp3D(inDim, outDim, factorT: temperalUpsample ? 2 : 1, factorS: 2) : nil
    }

    public func callAsFunction(_ x: MLXArray, firstChunk: Bool = false) -> MLXArray {
        var xMain = x
        for module in upsamples {
            switch module {
            case let r as V22Resample: xMain = r(xMain, firstChunk: firstChunk)
            case let rb as V22ResidualBlock: xMain = rb(xMain)
            default: fatalError("unexpected up-block: \(type(of: module))")
            }
            eval(xMain)  // bound graph size per sub-block
        }
        if let sc = avgShortcut {
            let xShort = sc(x, firstChunk: firstChunk)
            eval(xShort)
            return xMain + xShort
        }
        return xMain
    }
}

// MARK: - V22Head22 (decoder output head)

/// vae22.py `Head22` — RMS_norm → SiLU → CausalConv3d(dim→12). Keys `layer_0`/`layer_2`.
public final class V22Head22: Module, @unchecked Sendable {
    @ModuleInfo(key: "layer_0") public var layer0: V22RMSNorm
    @ModuleInfo(key: "layer_2") public var layer2: V22CausalConv3d

    public init(_ dim: Int, outChannels: Int = 12) {
        self._layer0.wrappedValue = V22RMSNorm(dim)
        self._layer2.wrappedValue = V22CausalConv3d(dim, outChannels, 3, padding: 1)
        super.init()
    }

    public func callAsFunction(
        _ x0: MLXArray, featCache: FeatCache? = nil, featIdx: FeatIdx? = nil
    ) -> MLXArray {
        var x = silu(layer0(x0))
        if let fc = featCache, let fi = featIdx {
            let idx = fi.value
            var cacheX = x[0..., Swift.max(0, x.dim(1) - CACHE_T)...]
            if cacheX.dim(1) < 2, let cached = fc[idx] {
                cacheX = concatenated([cached[0..., (cached.dim(1) - 1)...], cacheX], axis: 1)
            }
            x = layer2(x, cacheX: fc[idx])
            fc[idx] = cacheX
            fi.value += 1
        } else {
            x = layer2(x)
        }
        return x
    }
}

// MARK: - V22Decoder3d

/// vae22.py `Decoder3d` — conv1 → [ResBlock, AttnBlock, ResBlock] middle → 4 up stages
/// → Head22. dims = [dim·mult[-1]] + [dim·m for m in reversed(mult)] = [1024,1024,1024,512,256].
public final class V22Decoder3d: Module, @unchecked Sendable {
    @ModuleInfo(key: "conv1") public var conv1: V22CausalConv3d
    public let middle: [Module]
    public let upsamples: [Module]
    @ModuleInfo(key: "head") public var head: V22Head22

    public init(
        dim: Int = 256, zDim: Int = 48, dimMult: [Int] = [1, 2, 4, 4],
        numResBlocks: Int = 2, temperalUpsample: [Bool] = [true, true, false]
    ) {
        let dims = [dim * dimMult.last!] + dimMult.reversed().map { dim * $0 }
        self._conv1.wrappedValue = V22CausalConv3d(zDim, dims[0], 3, padding: 1)
        self.middle = [
            V22ResidualBlock(dims[0], dims[0]),
            V22AttentionBlock(dims[0]),
            V22ResidualBlock(dims[0], dims[0]),
        ]
        var ups = [Module]()
        for i in 0..<(dims.count - 1) {
            let tUp = i < temperalUpsample.count ? temperalUpsample[i] : false
            ups.append(
                V22UpResidualBlock(
                    inDim: dims[i], outDim: dims[i + 1], numResBlocks: numResBlocks + 1,
                    temperalUpsample: tUp, upFlag: i != dimMult.count - 1))
        }
        self.upsamples = ups
        self._head.wrappedValue = V22Head22(dims.last!)
        super.init()
    }

    /// x: [B, T, H, W, z_dim].
    public func callAsFunction(_ x0: MLXArray, firstChunk: Bool = false) -> MLXArray {
        var x = conv1(x0)
        for layer in middle {
            switch layer {
            case let r as V22ResidualBlock: x = r(x)
            case let a as V22AttentionBlock: x = a(x)
            default: fatalError("unexpected middle layer: \(type(of: layer))")
            }
        }
        eval(x)
        for layer in upsamples {
            x = (layer as! V22UpResidualBlock)(x, firstChunk: firstChunk)
            eval(x)
        }
        return head(x)
    }
}

// MARK: - Wan22VAEDecoder (full decode: conv2 → Decoder3d → unpatchify → clip)

/// vae22.py `Wan22VAEDecoder` — the 48-ch channels-last decode path. Input `z` must
/// already be denormalized (`denormalizeLatents22`). Output [B,T',H',W',3] in [-1,1],
/// T' = (T_lat-1)·4+1 (E11 first-chunk temporal trim).
public final class Wan22VAEDecoder: Module, @unchecked Sendable {
    public let zDim: Int
    @ModuleInfo(key: "conv2") public var conv2: V22CausalConv3d
    @ModuleInfo(key: "decoder") public var decoder: V22Decoder3d

    public init(zDim: Int = 48, dim: Int = 160, decDim: Int = 256) {
        self.zDim = zDim
        self._conv2.wrappedValue = V22CausalConv3d(zDim, zDim, 1)
        self._decoder.wrappedValue = V22Decoder3d(
            dim: decDim, zDim: zDim, dimMult: [1, 2, 4, 4], numResBlocks: 2,
            temperalUpsample: [true, true, false])
        super.init()
    }

    /// z: [B,T,H,W,48] (denormalized). → video [B,T',H',W',3] in [-1,1].
    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        let x = conv2(z)
        let out = decoder(x, firstChunk: true)
        return clip(unpatchify22(out), min: MLXArray(Float(-1)), max: MLXArray(Float(1)))
    }
}
