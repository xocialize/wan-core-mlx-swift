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

/// Streaming upsample3d reproducing the corrected whole-seq first-chunk skip
/// (E11) BIT-IDENTICALLY, chunk-by-chunk, via the 3-state `Rep` cache.
///
/// Whole-seq `Resample.upsample3d` (featCache==nil) bypasses time_conv for
/// frame 0 of the first chunk and zero-pads the `rest` (-> 2T-1, not 2T). To
/// match that streaming-wise:
///   - `first` chunk, chunk-local T>1: do the bypass (frame 0 passes through,
///     rest gets time_conv with a zero-padded causal start); cache the rest tail.
///   - `first` chunk, T==1, `single` (whole latent is one frame, T_lat==1):
///     mirror whole-seq's else branch — DOUBLE the lone frame (zero-pad).
///   - `first` chunk, T==1, more chunks follow: mark `Rep` — frame 0 is NOT
///     doubled and the NEXT chunk must zero-pad (treat its frame as the fresh
///     start of `rest`), so don't seed the cache with frame 0.
///   - later chunks: standard cached time_conv, except a `Rep` predecessor
///     forces a zero-pad (cacheX=nil) — the first `rest` frame after the skip.
private func resampleUpsample3dCached(
    _ rs: Resample, _ x: MLXArray, _ fc: FeatCache, _ fi: FeatIdx,
    first: Bool, single: Bool
) -> MLXArray {
    var x = x
    let (b, c, h, w) = (x.dim(0), x.dim(1), x.dim(3), x.dim(4))
    let t = x.dim(2)
    let idx = fi.value

    if first {
        if t > 1 {
            // frame 0 bypasses time_conv; rest = x[:,:,1:] gets a zero-padded
            // causal start (cacheX=nil) -> 1 + (T-1)*2 = 2T-1.
            let firstFrame = x[0..., 0..., 0..<1]
            let rest = x[0..., 0..., 1...]
            let tc = rs.timeConv!(rest)
            let restUp = temporalInterleave(tc, b, c, t - 1, h, w)
            x = concatenated([firstFrame, restUp], axis: 2)
            // cache for the next chunk = last CACHE_T frames of the conv INPUT (rest).
            fc[idx] = rest[0..., 0..., Swift.max(0, rest.dim(2) - CACHE_T)...]
        } else if single {
            // whole latent is one frame: whole-seq's `t>1` test is false, so it
            // doubles the lone frame. No chunk follows; cache is moot.
            let boundary = x[0..., 0..., Swift.max(0, t - CACHE_T)...]
            let tc = rs.timeConv!(x)
            x = temporalInterleave(tc, b, c, t, h, w)
            fc[idx] = boundary
        } else {
            // single-frame first chunk, more chunks follow: bypass (no doubling),
            // next chunk zero-pads. x (one frame) passes straight to spatial.
            fc.setRep(idx)
        }
        fi.value += 1
    } else {
        let isRep = fc.isRep(idx)
        let prev = fc[idx]  // nil when Rep
        var cacheX = x[0..., 0..., Swift.max(0, t - CACHE_T)...]
        if cacheX.dim(2) < 2, !isRep, let prev {
            cacheX = concatenated([prev[0..., 0..., (prev.dim(2) - 1)...], cacheX], axis: 2)
        }
        if cacheX.dim(2) < 2, isRep {
            // Rep predecessor: the fresh `rest` start zero-pads, so the carried
            // cache is [0, frame] to feed the following chunk's causal window.
            cacheX = concatenated([MLXArray.zeros(like: cacheX), cacheX], axis: 2)
        }
        let cacheArg: MLXArray? = isRep ? nil : prev  // Rep -> zero-pad
        let tc = rs.timeConv!(x, cacheX: cacheArg)
        fc[idx] = cacheX
        fi.value += 1
        x = temporalInterleave(tc, b, c, t, h, w)
    }

    return spatialUpsample(rs, x)
}

/// One latent chunk through Decoder3d, threading the shared temporal cache.
/// `first` = global first chunk (drives the upsample3d frame-0 skip); `single`
/// = the whole latent is a single frame (T_lat==1), which whole-seq doubles.
private func decoderChunk(
    _ decoder: Decoder3d, _ x: MLXArray, _ fc: FeatCache, _ fi: FeatIdx,
    first: Bool, single: Bool
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
                x = resampleUpsample3dCached(layer, x, fc, fi, first: first, single: single)
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
    // T_lat==1: whole-seq decode doubles the lone latent frame at the first
    // upsample3d (its `t>1` bypass test is false), so the single chunk must
    // double too rather than `Rep`-defer (which only suits a multi-chunk stream).
    let single = tLat == 1
    let fc = FeatCache(count: 64)  // generous; one slot per cached conv
    var outs: [MLXArray] = []
    var start = 0
    while start < tLat {
        let first = start == 0
        let zc = z[0..., 0..., start..<Swift.min(start + chunkLat, tLat)]
        let xc = vae.conv2(zc)  // kernel-1, per-frame, no cache
        let fi = FeatIdx()
        // Per-chunk visibility: region times this chunk (bounded by the eval below) and stamps
        // active/cache/phys → `[WANPROF] decode,chunk,N,ms,…`. One run tells us whether per-chunk
        // wall-time and phys are flat (streaming working) or growing (reclaim broken on this stream),
        // separating "CPU is just slow" from "streaming isn't streaming." No-op when profiling off.
        let oc = WanProfiler.shared.region("decode", "chunk", index: start) { () -> MLXArray in
            let oc = decoderChunk(vae.decoder, xc, fc, fi, first: first, single: single)
            // Materialize the carried cache too — else fc holds lazy slice-views
            // into this chunk's freed buffers, which alias/go stale across the
            // boundary (fails at >2 chunks).
            eval([oc] + fc.slots.compactMap { $0 })
            return oc
        }
        outs.append(oc)
        start += chunkLat
    }
    return clip(concatenated(outs, axis: 2), min: -1, max: 1)
}
