// 1:1 translation of bernini_r_mlx/streaming_decode.py — lossless streaming
// (temporal-chunked) VAE decode. Whole-sequence decode peaks ~2.3 GB/frame
// and OOMs past ~49 frames; this decodes ONE temporal chunk at a time,
// threading the decoder's CausalConv3d cross-chunk cache exactly like the
// chunked encode, so peak memory is flat in length and the output is
// BIT-IDENTICAL to `vae.decode(z)`.
//
// Consumer-side extension (no WanVAE edits): ResidualBlock already threads
// featCache; the two gaps — the upsample3d Resample (its stock path always
// re-doubles without cross-chunk cache) and the top-level chunk loop — are
// filled here. NCHWD (B,C,T,H,W); Phase-1 temporal only (spatial halo-tile
// deferred), matching the oracle.

import Foundation
import MLX
import MLXNN

/// CausalConv3d with cross-chunk temporal cache (mirrors ResidualBlock).
private func convCached(
    _ conv: CausalConv3d, _ x: MLXArray, _ fc: FeatCache, _ fi: FeatIdx
) -> MLXArray {
    let idx = fi.value
    var cacheX = x[0..., 0..., Swift.max(0, x.dim(2) - CACHE_T)...]
    if cacheX.dim(2) < 2, let cached = fc[idx] {
        cacheX = concatenated([cached[0..., 0..., (cached.dim(2) - 1)...], cacheX], axis: 2)
    }
    let out = conv(x, cacheX: fc[idx])
    fc[idx] = cacheX
    fi.value += 1
    return out
}

/// mlx-video upsample3d channel->frame interleave: [B,2C,T,H,W] -> [B,C,2T,H,W].
private func temporalInterleave(
    _ tc: MLXArray, _ b: Int, _ c: Int, _ t: Int, _ h: Int, _ w: Int
) -> MLXArray {
    let r = tc.reshaped(b, 2, c, t, h, w)
    return stacked([r[0..., 0], r[0..., 1]], axis: 3).reshaped(b, c, t * 2, h, w)
}

/// Per-frame nearest-2x + Conv2d (identical to stock; no temporal mixing).
private func spatialUpsample(_ rs: Resample, _ x: MLXArray) -> MLXArray {
    let (b, c, t, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
    var xf = x.transposed(0, 2, 3, 4, 1).reshaped(b * t, h, w, c)  // [BT,H,W,C]
    xf = repeated(xf, count: 2, axis: 1)
    xf = repeated(xf, count: 2, axis: 2)
    xf = rs.resample[1]!(xf)  // Conv2d -> [BT,2H,2W,C//2]
    let co = xf.dim(3)
    return xf.reshaped(b, t, h * 2, w * 2, co).transposed(0, 4, 1, 2, 3)
}

/// Streaming upsample3d: cached `time_conv` (causal zero-pad on the first
/// chunk, prev-chunk tail thereafter) + interleave + spatial. mlx-video's
/// stock upsample3d always doubles every frame — no first-chunk frame-0 skip.
private func resampleUpsample3dCached(
    _ rs: Resample, _ x: MLXArray, _ fc: FeatCache, _ fi: FeatIdx
) -> MLXArray {
    let (b, c, t, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
    let idx = fi.value
    var cacheX = x[0..., 0..., Swift.max(0, x.dim(2) - CACHE_T)...]
    if cacheX.dim(2) < 2, let cached = fc[idx] {
        cacheX = concatenated([cached[0..., 0..., (cached.dim(2) - 1)...], cacheX], axis: 2)
    }
    let tc = rs.timeConv!(x, cacheX: fc[idx])  // fc[idx]=nil first chunk -> zero-pad
    fc[idx] = cacheX
    fi.value += 1
    let interleaved = temporalInterleave(tc, b, c, t, h, w)
    return spatialUpsample(rs, interleaved)
}

/// One latent chunk through Decoder3d, threading the shared temporal cache.
private func decoderChunk(
    _ decoder: Decoder3d, _ x: MLXArray, _ fc: FeatCache, _ fi: FeatIdx
) -> MLXArray {
    var x = convCached(decoder.conv1, x, fc, fi)
    for layer in decoder.middle {
        switch layer {
        case let layer as AttentionBlock: x = layer(x)
        case let layer as ResidualBlock: x = layer(x, featCache: fc, featIdx: fi)
        default: fatalError("unexpected middle layer: \(type(of: layer))")
        }
    }
    for layer in decoder.upsamples {
        switch layer {
        case let layer as ResidualBlock:
            x = layer(x, featCache: fc, featIdx: fi)
        case let layer as Resample:
            if layer.mode == "upsample3d" {
                x = resampleUpsample3dCached(layer, x, fc, fi)
            } else {  // upsample2d: per-frame, no temporal
                x = layer(x)
            }
        default: fatalError("unexpected upsample layer: \(type(of: layer))")
        }
    }
    x = silu((decoder.head[0] as! RMS_norm)(x))  // head: RMS_norm, silu, CausalConv3d
    x = convCached(decoder.head[2] as! CausalConv3d, x, fc, fi)
    return x
}

/// Lossless temporal-chunked decode. Bit-identical to `vae.decode(z)`, flat
/// peak memory. z: normalized latent [B, zDim, T_lat, H, W] ->
/// video [B, 3, T_out, H, W] in [-1, 1].
public func decodeStreaming(
    vae: WanVAE, _ z: MLXArray, chunkLat: Int = 1
) -> MLXArray {
    let mean = vae.mean.reshaped(1, -1, 1, 1, 1)
    let invStd = vae.invStd.reshaped(1, -1, 1, 1, 1)
    let z = z / invStd + mean

    let tLat = z.dim(2)
    let fc = FeatCache(count: 64)  // generous; one slot per cached conv
    var outs: [MLXArray] = []
    var start = 0
    while start < tLat {
        let zc = z[0..., 0..., start..<Swift.min(start + chunkLat, tLat)]
        let xc = vae.conv2(zc)  // kernel-1, per-frame, no cache
        let fi = FeatIdx()
        let oc = decoderChunk(vae.decoder, xc, fc, fi)
        // Materialize the carried cache too — else fc holds lazy slice-views
        // into this chunk's freed buffers, which alias/go stale across the
        // boundary (fails at >2 chunks).
        eval([oc] + fc.slots.compactMap { $0 })
        outs.append(oc)
        start += chunkLat
    }
    return clip(concatenated(outs, axis: 2), min: -1, max: 1)
}
