// WanVAE.swift — 3D VAE for Wan2.1/2.2 (compression 4×8×8).
//
// 1:1 port of mlx-video `wan_2/vae.py` (@87db56a). Same class names, same
// method decomposition, same forward-pass order — a side-by-side diff against
// vae.py should show only Python↔Swift substitutions. Module structure mirrors
// the original PyTorch checkpoint key hierarchy so weights load directly
// without key sanitization (the 194-key contract in
// Tests/BerniniRTests/Fixtures/vae_keys.txt).
//
// Swift-idiom substitutions (numerics-neutral, donor-proven in
// longcat-avatar-mlx-swift AutoencoderKLWan.swift):
//   - `CausalConv3d._conv3d` keeps vae.py's method decomposition, but uses
//     native `MLX.conv3d` instead of Python-MLX's per-timestep Conv2d
//     emulation (Python MLX historically lacked Conv3d; weight layout
//     `(O, kT, kH, kW, I)` is identical in both runtimes).
//   - The mid-block AttentionBlock routes its SDPA through the `.cpu` stream
//     for strict fp32 (donor lesson L10 — Metal loses ~3-4 bits of fp32
//     precision on the QK^T/softmax-V chain; the VAE has only two
//     AttentionBlocks so the perf cost is negligible).
//   - vae.py's `feat_cache` (list of Optional arrays) / `feat_idx` ([0] int
//     box) become the `FeatCache` / `FeatIdx` reference wrappers.
//   - vae.py's `mean` / `std` / `inv_std` are `mx.array` attributes, which
//     Python MLX counts as module parameters (mlx-video loads with
//     strict=False to skip them). Swift stores them as computed properties so
//     the flattened parameter key set is exactly the 194 checkpoint keys and
//     we can load strictly.
//
// Deliberate omission: vae.py's `decode_tiled` is a thin wrapper over
// `wan_2/tiling.py` (LTX-2 tiling infrastructure, a separate 338-LOC module
// outside this component). The Bernini oracle never calls it — memory-bounded
// decode is the S5 `streaming_decode` port per PORTING-SPEC.md.
//
// The checkpoint VAE is fp32 — all math here stays fp32.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Python `CACHE_T`: trailing temporal frames each cached CausalConv3d keeps
/// to give the next chunk its causal context.
public let CACHE_T = 2

/// Per-channel normalization statistics for z_dim=16 (constants in vae.py —
/// the checkpoint config.json carries no latent stats).
public let VAE_MEAN: [Float] = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
public let VAE_STD: [Float] = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
]

// MARK: - chunked-encode state (Python: feat_cache list + feat_idx [0] box)

/// Reference wrapper for Python's `feat_cache` — a mutable list of
/// Optional cached tensors, indexed by conv slot.
public final class FeatCache: @unchecked Sendable {
    public var slots: [MLXArray?]

    public init(count: Int) {
        self.slots = Array(repeating: nil, count: count)
    }

    public subscript(idx: Int) -> MLXArray? {
        get { slots[idx] }
        set { slots[idx] = newValue }
    }
}

/// Reference wrapper for Python's `feat_idx = [0]` single-int box.
public final class FeatIdx: @unchecked Sendable {
    public var value: Int = 0

    public init() {}
}

// MARK: - CausalConv3d

/// 3D convolution with causal temporal padding.
public final class CausalConv3d: Module, @unchecked Sendable {
    public let kernelSize: (Int, Int, Int)
    public let stride: (Int, Int, Int)
    let causalPadT: Int
    let padH: Int
    let padW: Int

    // MLX Conv3d: weight shape [O, D, H, W, I]
    public let weight: MLXArray
    public let bias: MLXArray

    public init(
        _ inChannels: Int,
        _ outChannels: Int,
        _ kernelSize: (Int, Int, Int),
        stride: (Int, Int, Int) = (1, 1, 1),
        padding: (Int, Int, Int) = (0, 0, 0)
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        // Causal padding: match reference formula dilation*(k-1) + (1-stride)
        // With dilation=1: k-stride (pads left only, no future context)
        self.causalPadT = kernelSize.0 - stride.0
        self.padH = padding.1
        self.padW = padding.2

        self.weight = MLXArray.zeros(
            [outChannels, kernelSize.0, kernelSize.1, kernelSize.2, inChannels]
        )
        self.bias = MLXArray.zeros([outChannels])
        super.init()
    }

    /// Python's scalar kernel_size / padding spelling.
    public convenience init(_ inChannels: Int, _ outChannels: Int, _ kernelSize: Int, padding: Int = 0) {
        self.init(
            inChannels, outChannels, (kernelSize, kernelSize, kernelSize),
            padding: (padding, padding, padding)
        )
    }

    /// x: [B, C, T, H, W] (channel-first)
    public func callAsFunction(_ x: MLXArray, cacheX: MLXArray? = nil) -> MLXArray {
        var x = x
        let (b, c, _, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))

        var causalPad = causalPadT
        if let cacheX, causalPad > 0 {
            x = concatenated([cacheX, x], axis: 2)
            causalPad = Swift.max(0, causalPad - cacheX.dim(2))
        }

        if causalPad > 0 {
            let padT = MLXArray.zeros([b, c, causalPad, h, w], dtype: x.dtype)
            x = concatenated([padT, x], axis: 2)
        }

        if padH > 0 || padW > 0 {
            x = padded(
                x,
                widths: [
                    .init((0, 0)),
                    .init((0, 0)),
                    .init((0, 0)),
                    .init((padH, padH)),
                    .init((padW, padW)),
                ]
            )
        }

        x = x.transposed(0, 2, 3, 4, 1)  // [B, T, H, W, C]
        let out = _conv3d(x)
        return out.transposed(0, 4, 1, 2, 3)  // [B, O, T', H', W']
    }

    /// 3D conv core. vae.py emulates this with a sliding window of 2D convs
    /// (Python MLX lacked Conv3d); mlx-swift has native `conv3d` with the
    /// identical `[O, D, H, W, I]` weight layout, so we call it directly.
    /// x: [B, T, H, W, C_in] -> [B, T_out, H_out, W_out, C_out]
    private func _conv3d(_ x: MLXArray) -> MLXArray {
        conv3d(x, weight, stride: .init((stride.0, stride.1, stride.2)), padding: 0) + bias
    }
}

// MARK: - RMS_norm

/// Channel-first L2 normalization matching original Wan VAE.
///
/// Uses F.normalize (L2 norm) with learned scale, equivalent to RMS norm.
/// images=true: gamma shape (dim, 1, 1) for 4D (per-frame) input.
/// images=false: gamma shape (dim, 1, 1, 1) for 5D video input.
public final class RMS_norm: Module, @unchecked Sendable {
    public let channelFirst: Bool
    public let scale: Float
    public let gamma: MLXArray

    public init(_ dim: Int, channelFirst: Bool = true, images: Bool = true) {
        self.channelFirst = channelFirst
        self.scale = Float(pow(Double(dim), 0.5))
        if channelFirst {
            let broadcastable = images ? [1, 1] : [1, 1, 1]
            self.gamma = MLXArray.ones([dim] + broadcastable)
        } else {
            self.gamma = MLXArray.ones([dim])
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normDim = channelFirst ? 1 : -1
        // L2 normalize along channel dim (matches F.normalize)
        let norm = sqrt(
            clip(sum(x * x, axis: normDim, keepDims: true), min: 1e-12)
        )
        return (x / norm) * scale * gamma
    }
}

// MARK: - ResidualBlock

/// Residual block with causal 3D convolutions.
///
/// Uses `residual` list with nil gaps to match original PyTorch
/// nn.Sequential indices: [0]=norm, [1]=SiLU, [2]=conv, [3]=norm,
/// [4]=SiLU, [5]=Dropout, [6]=conv. Only indices 0,2,3,6 have params.
public final class ResidualBlock: Module, @unchecked Sendable {
    public let residual: [Module?]
    public let shortcut: CausalConv3d?

    // Typed views of the heterogeneous `residual` list (computed properties
    // are invisible to module reflection — keys stay residual.0/.2/.3/.6).
    private var residual0: RMS_norm { residual[0] as! RMS_norm }
    private var residual2: CausalConv3d { residual[2] as! CausalConv3d }
    private var residual3: RMS_norm { residual[3] as! RMS_norm }
    private var residual6: CausalConv3d { residual[6] as! CausalConv3d }

    public init(_ inDim: Int, _ outDim: Int) {
        self.residual = [
            RMS_norm(inDim, images: false),  // [0]
            nil,  // [1] SiLU
            CausalConv3d(inDim, outDim, 3, padding: 1),  // [2]
            RMS_norm(outDim, images: false),  // [3]
            nil,  // [4] SiLU
            nil,  // [5] Dropout
            CausalConv3d(outDim, outDim, 3, padding: 1),  // [6]
        ]
        self.shortcut = inDim != outDim ? CausalConv3d(inDim, outDim, 1) : nil
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, featCache: FeatCache? = nil, featIdx: FeatIdx? = nil
    ) -> MLXArray {
        var x = x
        let h = shortcut == nil ? x : shortcut!(x)

        if let featCache, let featIdx {
            // First conv: norm -> silu -> [cache] -> conv
            x = silu(residual0(x))
            var idx = featIdx.value
            var cacheX = x[0..., 0..., Swift.max(0, x.dim(2) - CACHE_T)...]
            if cacheX.dim(2) < CACHE_T, let cached = featCache[idx] {
                cacheX = concatenated([cached[0..., 0..., (cached.dim(2) - 1)...], cacheX], axis: 2)
            }
            x = residual2(x, cacheX: featCache[idx])
            featCache[idx] = cacheX
            featIdx.value += 1

            // Second conv: norm -> silu -> [cache] -> conv
            x = silu(residual3(x))
            idx = featIdx.value
            cacheX = x[0..., 0..., Swift.max(0, x.dim(2) - CACHE_T)...]
            if cacheX.dim(2) < CACHE_T, let cached = featCache[idx] {
                cacheX = concatenated([cached[0..., 0..., (cached.dim(2) - 1)...], cacheX], axis: 2)
            }
            x = residual6(x, cacheX: featCache[idx])
            featCache[idx] = cacheX
            featIdx.value += 1
        } else {
            x = silu(residual0(x))
            x = residual2(x)
            x = silu(residual3(x))
            x = residual6(x)
        }

        return x + h
    }
}

// MARK: - AttentionBlock

/// Single-head spatial self-attention.
public final class AttentionBlock: Module, @unchecked Sendable {
    public let norm: RMS_norm
    @ModuleInfo(key: "to_qkv") public var toQKV: Conv2d
    public let proj: Conv2d

    public init(_ dim: Int) {
        self.norm = RMS_norm(dim, images: true)
        self._toQKV.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim * 3, kernelSize: 1)
        self.proj = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
        super.init()
    }

    /// x: [B, C, T, H, W]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let identity = x
        let (b, c, t, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))

        // [B,C,T,H,W] -> [B,T,C,H,W] -> [BT,C,H,W] -> norm -> [BT,H,W,C]
        var x = x.transposed(0, 2, 1, 3, 4).reshaped(b * t, c, h, w)
        x = norm(x)
        x = x.transposed(0, 2, 3, 1)  // [BT, H, W, C]

        let qkv = toQKV(x)  // [BT, H, W, 3C]
        let qkvSplit = qkv.reshaped(b * t, h * w, 3, c).transposed(2, 0, 1, 3)
        let (q, k, v) = (qkvSplit[0], qkvSplit[1], qkvSplit[2])

        // [BT, 1, HW, C] — donor lesson L10: run SDPA on the .cpu stream for
        // strict fp32 (Metal loses ~3-4 bits on this chain).
        var out = MLXFast.scaledDotProductAttention(
            queries: q[0..., .newAxis, 0..., 0...],
            keys: k[0..., .newAxis, 0..., 0...],
            values: v[0..., .newAxis, 0..., 0...],
            scale: Float(pow(Double(c), -0.5)),
            mask: nil,
            stream: .cpu
        )
        // materialize before leaving the cpu stream (donor idiom)
        out.eval()
        out = out.squeezed(axis: 1).reshaped(b * t, h, w, c)  // [BT, H, W, C]

        out = proj(out)  // [BT, H, W, C]
        out = out.reshaped(b, t, h, w, c).transposed(0, 4, 1, 2, 3)  // [B, C, T, H, W]
        return out + identity
    }
}

// MARK: - Resample

/// Resample block matching original Wan VAE structure.
///
/// Supports both upsampling (decoder) and downsampling (encoder).
/// Uses list-based param storage to match original nn.Sequential key hierarchy.
public final class Resample: Module, @unchecked Sendable {
    public let mode: String
    public let dim: Int

    // resample.0 = Upsample/ZeroPad2d (no params), resample.1 = Conv2d
    public let resample: [Conv2d?]
    @ModuleInfo(key: "time_conv") public var timeConv: CausalConv3d?

    private var resample1: Conv2d { resample[1]! }

    public init(_ dim: Int, mode: String) {
        precondition(
            ["upsample2d", "upsample3d", "downsample2d", "downsample3d"].contains(mode)
        )
        self.mode = mode
        self.dim = dim

        if mode.hasPrefix("upsample") {
            // resample.0 = Upsample (no params), resample.1 = Conv2d
            self.resample = [
                nil, Conv2d(inputChannels: dim, outputChannels: dim / 2, kernelSize: 3, padding: 1),
            ]
            self._timeConv.wrappedValue =
                mode == "upsample3d"
                ? CausalConv3d(dim, dim * 2, (3, 1, 1), padding: (1, 0, 0))
                : nil
        } else {
            // resample.0 = ZeroPad2d (no params), resample.1 = Conv2d(stride=2)
            self.resample = [
                nil, Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, stride: 2),
            ]
            self._timeConv.wrappedValue =
                mode == "downsample3d"
                ? CausalConv3d(dim, dim, (3, 1, 1), stride: (2, 1, 1), padding: (0, 0, 0))
                : nil
        }
        super.init()
    }

    /// x: [B, C, T, H, W]
    public func callAsFunction(
        _ x: MLXArray, featCache: FeatCache? = nil, featIdx: FeatIdx? = nil
    ) -> MLXArray {
        var x = x
        let (b, c, h, w) = (x.dim(0), x.dim(1), x.dim(3), x.dim(4))
        var t = x.dim(2)

        if mode == "upsample3d" {
            // Temporal upsample via learned conv
            var xT = timeConv!(x)  // [B, 2C, T, H, W]
            xT = xT.reshaped(b, 2, c, t, h, w)
            x = stacked([xT[0..., 0], xT[0..., 1]], axis: 3).reshaped(b, c, t * 2, h, w)
            t = t * 2
        }

        if mode.hasPrefix("upsample") {
            // Per-frame spatial upsample: nearest 2x + Conv2d
            x = x.transposed(0, 2, 3, 4, 1).reshaped(b * t, h, w, c)  // [BT, H, W, C]
            x = repeated(x, count: 2, axis: 1)
            x = repeated(x, count: 2, axis: 2)
            x = resample1(x)  // Conv2d [BT, 2H, 2W, C//2]
            let cOut = x.dim(-1)
            return x.reshaped(b, t, h * 2, w * 2, cOut).transposed(0, 4, 1, 2, 3)
        } else {
            // Per-frame spatial downsample: ZeroPad(0,1,0,1) + Conv2d(stride=2)
            x = x.transposed(0, 2, 3, 4, 1).reshaped(b * t, h, w, c)  // [BT, H, W, C]
            x = padded(x, widths: [.init((0, 0)), .init((0, 1)), .init((0, 1)), .init((0, 0))])  // ZeroPad2d(0,1,0,1)
            x = resample1(x)  // Conv2d stride=2
            let cOut = x.dim(-1)
            let (hOut, wOut) = (x.dim(1), x.dim(2))
            x = x.reshaped(b, t, hOut, wOut, cOut).transposed(0, 4, 1, 2, 3)

            if mode == "downsample3d" {
                if let featCache, let featIdx {
                    let idx = featIdx.value
                    if featCache[idx] == nil {
                        // First chunk: save x, skip time_conv
                        featCache[idx] = x
                        featIdx.value += 1
                    } else {
                        // Subsequent chunks: use cached frame as temporal context
                        let cacheX = x[0..., 0..., (x.dim(2) - 1)...]
                        let cached = featCache[idx]!
                        x = timeConv!(x, cacheX: cached[0..., 0..., (cached.dim(2) - 1)...])
                        featCache[idx] = cacheX
                        featIdx.value += 1
                    }
                } else {
                    x = timeConv!(x)
                }
            }
            return x
        }
    }
}

// MARK: - Decoder3d

/// 3D VAE Decoder matching Wan2.1 architecture.
///
/// Uses flat `middle` and `upsamples` lists to match original
/// PyTorch nn.Sequential weight key hierarchy.
public final class Decoder3d: Module, @unchecked Sendable {
    public let conv1: CausalConv3d
    public let middle: [Module]
    public let upsamples: [Module]
    public let head: [Module?]

    private var head0: RMS_norm { head[0] as! RMS_norm }
    private var head2: CausalConv3d { head[2] as! CausalConv3d }

    public init(
        dim: Int = 96,
        zDim: Int = 16,
        dimMult: [Int] = [1, 2, 4, 4],
        numResBlocks: Int = 2,
        temporalUpsample: [Bool] = [true, true, false]
    ) {
        let dims = ([dimMult.last!] + dimMult.reversed()).map { dim * $0 }

        self.conv1 = CausalConv3d(zDim, dims[0], 3, padding: 1)

        // Middle: [ResBlock, AttentionBlock, ResBlock]
        self.middle = [
            ResidualBlock(dims[0], dims[0]),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0]),
        ]

        // Flat upsample list matching original nn.Sequential indexing
        var upsamples = [Module]()
        for (i, (inDimRaw, outDim)) in zip(dims.dropLast(), dims.dropFirst()).enumerated() {
            var inDim = inDimRaw
            if [1, 2, 3].contains(i) {
                inDim = inDim / 2
            }
            for _ in 0..<(numResBlocks + 1) {
                upsamples.append(ResidualBlock(inDim, outDim))
                inDim = outDim
            }
            if i != dimMult.count - 1 {
                let mode = temporalUpsample[i] ? "upsample3d" : "upsample2d"
                upsamples.append(Resample(outDim, mode: mode))
            }
        }
        self.upsamples = upsamples

        // Output head: [RMS_norm, SiLU (no params), CausalConv3d]
        self.head = [
            RMS_norm(dims.last!, images: false),  // [0]
            nil,  // [1] SiLU
            CausalConv3d(dims.last!, 3, 3, padding: 1),  // [2]
        ]
        super.init()
    }

    /// x: [B, z_dim, T, H, W] -> [B, 3, T_out, H_out, W_out]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = conv1(x)

        for layer in middle {
            switch layer {
            case let layer as ResidualBlock: x = layer(x)
            case let layer as AttentionBlock: x = layer(x)
            default: fatalError("unexpected middle layer: \(type(of: layer))")
            }
        }

        for layer in upsamples {
            switch layer {
            case let layer as ResidualBlock: x = layer(x)
            case let layer as Resample: x = layer(x)
            default: fatalError("unexpected upsample layer: \(type(of: layer))")
            }
        }

        x = silu(head0(x))
        x = head2(x)
        return x
    }
}

// MARK: - Encoder3d

/// 3D VAE Encoder matching Wan2.1 architecture.
///
/// Mirror of Decoder3d with downsampling instead of upsampling.
/// Uses flat lists to match original PyTorch nn.Sequential weight key hierarchy.
public final class Encoder3d: Module, @unchecked Sendable {
    public let conv1: CausalConv3d
    public let downsamples: [Module]
    public let middle: [Module]
    public let head: [Module?]

    private var head0: RMS_norm { head[0] as! RMS_norm }
    private var head2: CausalConv3d { head[2] as! CausalConv3d }

    public init(
        dim: Int = 96,
        zDim: Int = 16,
        dimMult: [Int] = [1, 2, 4, 4],
        numResBlocks: Int = 2,
        temporalDownsample: [Bool] = [false, true, true]
    ) {
        let dims = ([1] + dimMult).map { dim * $0 }

        self.conv1 = CausalConv3d(3, dims[0], 3, padding: 1)

        // Flat downsample list matching original nn.Sequential indexing
        var downsamples = [Module]()
        for (i, (inDimRaw, outDim)) in zip(dims.dropLast(), dims.dropFirst()).enumerated() {
            var inDim = inDimRaw
            for _ in 0..<numResBlocks {
                downsamples.append(ResidualBlock(inDim, outDim))
                inDim = outDim
            }
            if i != dimMult.count - 1 {
                let mode = temporalDownsample[i] ? "downsample3d" : "downsample2d"
                downsamples.append(Resample(outDim, mode: mode))
            }
        }
        self.downsamples = downsamples

        // Middle: [ResBlock, AttentionBlock, ResBlock]
        self.middle = [
            ResidualBlock(dims.last!, dims.last!),
            AttentionBlock(dims.last!),
            ResidualBlock(dims.last!, dims.last!),
        ]

        // Output head: [RMS_norm, SiLU (no params), CausalConv3d]
        self.head = [
            RMS_norm(dims.last!, images: false),
            nil,  // SiLU
            CausalConv3d(dims.last!, zDim, 3, padding: 1),
        ]
        super.init()
    }

    /// x: [B, 3, T, H, W] -> [B, z_dim, T_lat, H_lat, W_lat]
    public func callAsFunction(
        _ x: MLXArray, featCache: FeatCache? = nil, featIdx: FeatIdx? = nil
    ) -> MLXArray {
        var x = x
        if let featCache, let featIdx {
            // conv1 with caching
            let idx = featIdx.value
            var cacheX = x[0..., 0..., Swift.max(0, x.dim(2) - CACHE_T)...]
            if cacheX.dim(2) < CACHE_T, let cached = featCache[idx] {
                cacheX = concatenated([cached[0..., 0..., (cached.dim(2) - 1)...], cacheX], axis: 2)
            }
            x = conv1(x, cacheX: featCache[idx])
            featCache[idx] = cacheX
            featIdx.value += 1
        } else {
            x = conv1(x)
        }

        for layer in downsamples {
            switch layer {
            case let layer as ResidualBlock: x = layer(x, featCache: featCache, featIdx: featIdx)
            case let layer as Resample: x = layer(x, featCache: featCache, featIdx: featIdx)
            default: fatalError("unexpected downsample layer: \(type(of: layer))")
            }
        }

        for layer in middle {
            switch layer {
            case let layer as ResidualBlock: x = layer(x, featCache: featCache, featIdx: featIdx)
            case let layer as AttentionBlock: x = layer(x)
            default: fatalError("unexpected middle layer: \(type(of: layer))")
            }
        }

        if let featCache, let featIdx {
            // Head: norm -> silu -> [cache] -> conv
            x = silu(head0(x))
            let idx = featIdx.value
            var cacheX = x[0..., 0..., Swift.max(0, x.dim(2) - CACHE_T)...]
            if cacheX.dim(2) < CACHE_T, let cached = featCache[idx] {
                cacheX = concatenated([cached[0..., 0..., (cached.dim(2) - 1)...], cacheX], axis: 2)
            }
            x = head2(x, cacheX: featCache[idx])
            featCache[idx] = cacheX
            featIdx.value += 1
        } else {
            x = silu(head0(x))
            x = head2(x)
        }

        return x
    }
}

// MARK: - WanVAE

/// Wan2.1 VAE wrapper with per-channel normalization.
///
/// Supports both encode (for I2V / editing surfaces) and decode (for all models).
public final class WanVAE: Module, @unchecked Sendable {
    public let zDim: Int

    /// Latent stats. Computed (not stored MLXArray) so they stay out of the
    /// parameter key set — vae.py's `mx.array` attributes are skipped at load
    /// via strict=False; Swift loads strictly against the 194-key contract.
    public var mean: MLXArray { MLXArray(VAE_MEAN) }
    public var std: MLXArray { MLXArray(VAE_STD) }
    public var invStd: MLXArray { 1.0 / std }

    public let conv2: CausalConv3d
    public let decoder: Decoder3d

    public let encoder: Encoder3d?
    public let conv1: CausalConv3d?

    public init(zDim: Int = 16, encoder: Bool = false) {
        self.zDim = zDim

        self.conv2 = CausalConv3d(zDim, zDim, 1)
        self.decoder = Decoder3d(dim: 96, zDim: zDim)

        if encoder {
            self.encoder = Encoder3d(dim: 96, zDim: zDim * 2)
            self.conv1 = CausalConv3d(zDim * 2, zDim * 2, 1)
        } else {
            self.encoder = nil
            self.conv1 = nil
        }
        super.init()
    }

    /// Encode video to normalized latent using chunked encoding.
    ///
    /// Uses chunked encoding with temporal caching to match reference behavior.
    /// First frame encoded alone, then 4-frame chunks with cached context.
    ///
    /// - Parameter x: Video [B, 3, T, H, W] in [-1, 1]
    /// - Returns: Normalized latent [B, z_dim, T_lat, H_lat, W_lat]
    public func encode(_ x: MLXArray) -> MLXArray {
        // Count cacheable CausalConv3d slots in encoder
        let numSlots = _countEncoderCacheSlots()
        let featCache = FeatCache(count: numSlots)

        let t = x.dim(2)
        let numChunks = 1 + (t - 1) / 4

        var out: MLXArray? = nil
        for i in 0..<numChunks {
            let featIdx = FeatIdx()
            let chunk: MLXArray
            if i == 0 {
                chunk = x[0..., 0..., ..<1]
            } else {
                chunk = x[0..., 0..., (1 + 4 * (i - 1))..<(1 + 4 * i)]
            }

            let chunkOut = encoder!(chunk, featCache: featCache, featIdx: featIdx)

            if out == nil {
                out = chunkOut
            } else {
                out = concatenated([out!, chunkOut], axis: 2)
            }
        }

        let mu = split(conv1!(out!), parts: 2, axis: 1)[0]

        // Normalize: (mu - mean) * inv_std
        let mean = self.mean.reshaped(1, -1, 1, 1, 1)
        let invStd = self.invStd.reshaped(1, -1, 1, 1, 1)
        return (mu - mean) * invStd
    }

    /// Count CausalConv3d that participate in chunked encoding cache.
    func _countEncoderCacheSlots() -> Int {
        var count = 1  // encoder.conv1
        for layer in encoder!.downsamples {
            if layer is ResidualBlock {
                count += 2  // two convs in residual path
            } else if let layer = layer as? Resample, layer.mode == "downsample3d" {
                count += 1  // time_conv
            }
        }
        for layer in encoder!.middle {
            if layer is ResidualBlock {
                count += 2
            }
        }
        count += 1  // encoder.head CausalConv3d
        return count
    }

    /// Decode latent to video.
    ///
    /// - Parameter z: Normalized latent [B, z_dim, T, H, W]
    /// - Returns: Video [B, 3, T_out, H_out, W_out] clamped to [-1, 1]
    public func decode(_ z: MLXArray) -> MLXArray {
        let mean = self.mean.reshaped(1, -1, 1, 1, 1)
        let invStd = self.invStd.reshaped(1, -1, 1, 1, 1)
        let z = z / invStd + mean

        let x = conv2(z)
        let out = decoder(x)
        return clip(out, min: -1, max: 1)
    }

    // vae.py also defines `decode_tiled(z, tiling_config)`, a thin wrapper
    // over mlx_video.models.wan_2.tiling (LTX-2 tiling infrastructure —
    // a separate module outside this component, never called by the Bernini
    // oracle). Memory-bounded decode arrives as the S5 streaming-decode port.
}
