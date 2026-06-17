// Lossless streaming (temporal-chunked) decode for the 48-ch channels-LAST vae22 —
// the channels-last analog of StreamingDecode.swift. Whole-sequence vae22 decode
// peaks ~27 GB PER LATENT FRAME (measured: 720p 5f = 2 latent frames → +54.8 GB),
// OOMing past a few frames; this decodes ONE latent chunk at a time, threading the
// decoder's causal cross-chunk cache so peak memory is FLAT in length and the output
// is BIT-IDENTICAL to `Wan22VAEDecoder(z)`.
//
// vae22 layout is channels-last [B,T,H,W,C] (T=axis 1, C=axis 4) — vs the 16-ch
// [B,C,T,H,W] — and its decoder upsamples are NESTED (`V22UpResidualBlock` = resblocks
// + Resample + a param-free `DupUp3D` shortcut), so the cache threads through the nest.
// The `V22ResidualBlock`/`V22Head22`/`V22CausalConv3d` primitives already cache; the two
// gaps — the `upsample3d` Resample (its stock path re-doubles without a cross-chunk
// cache) and the top-level chunk loop — are filled here. Temporal only (the decode cost
// is ~linear in LATENT frames; spatial tiling not needed — Xcode-agent confirmed).

import Foundation
import MLX
import MLXNN

/// V22CausalConv3d with cross-chunk temporal cache (channels-last, T=axis 1).
private func convCached22(
    _ conv: V22CausalConv3d, _ x: MLXArray, _ fc: FeatCache, _ fi: FeatIdx
) -> MLXArray {
    let idx = fi.value
    var cacheX = x[0..., Swift.max(0, x.dim(1) - CACHE_T)...]
    if cacheX.dim(1) < 2, let cached = fc[idx] {
        cacheX = concatenated([cached[0..., (cached.dim(1) - 1)...], cacheX], axis: 1)
    }
    let out = conv(x, cacheX: fc[idx])
    fc[idx] = cacheX
    fi.value += 1
    return out
}

/// time_conv channel→frame interleave: [B,T,H,W,2C] → [B,2T,H,W,C].
private func temporalInterleave22(
    _ tc: MLXArray, _ b: Int, _ t: Int, _ h: Int, _ w: Int, _ c: Int
) -> MLXArray {
    let r = tc.reshaped(b, t, h, w, 2, c)
    let s0 = r[0..., 0..., 0..., 0..., 0, 0...]
    let s1 = r[0..., 0..., 0..., 0..., 1, 0...]
    return stacked([s0, s1], axis: 2).reshaped(b, t * 2, h, w, c)
}

/// Per-frame nearest-2× + 3×3 Conv2d (identical to stock; no temporal mixing).
private func spatialUpsample22(_ rs: V22Resample, _ x: MLXArray) -> MLXArray {
    let (b, t, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
    var xf = x.reshaped(b * t, h, w, c)
    xf = repeated(repeated(xf, count: 2, axis: 1), count: 2, axis: 2)
    xf = padded(xf, widths: [.init((0, 0)), .init((1, 1)), .init((1, 1)), .init((0, 0))])
    xf = conv2d(xf, rs.resampleWeight) + rs.resampleBias
    return xf.reshaped(b, t, xf.dim(1), xf.dim(2), c)
}

/// Streaming upsample3d reproducing the whole-seq first-chunk frame-0 skip (E11)
/// BIT-IDENTICALLY, chunk-by-chunk, via the 3-state `Rep` cache. (See the 16-ch
/// `resampleUpsample3dCached` for the full case analysis — this is its channels-last
/// twin.)
private func resampleUpsample3dCached22(
    _ rs: V22Resample, _ x0: MLXArray, _ fc: FeatCache, _ fi: FeatIdx,
    first: Bool, single: Bool
) -> MLXArray {
    var x = x0
    let (b, h, w, c) = (x.dim(0), x.dim(2), x.dim(3), x.dim(4))
    let t = x.dim(1)
    let idx = fi.value

    if first {
        if t > 1 {
            let firstFrame = x[0..., 0..<1]
            let rest = x[0..., 1...]
            let tc = rs.timeConv!(rest)
            let restUp = temporalInterleave22(tc, b, t - 1, h, w, c)
            x = concatenated([firstFrame, restUp], axis: 1)
            fc[idx] = rest[0..., Swift.max(0, rest.dim(1) - CACHE_T)...]
        } else if single {
            let boundary = x[0..., Swift.max(0, t - CACHE_T)...]
            let tc = rs.timeConv!(x)
            x = temporalInterleave22(tc, b, t, h, w, c)
            fc[idx] = boundary
        } else {
            fc.setRep(idx)  // bypass; next chunk zero-pads
        }
        fi.value += 1
    } else {
        let isRep = fc.isRep(idx)
        let prev = fc[idx]  // nil when Rep
        var cacheX = x[0..., Swift.max(0, t - CACHE_T)...]
        if cacheX.dim(1) < 2, !isRep, let prev {
            cacheX = concatenated([prev[0..., (prev.dim(1) - 1)...], cacheX], axis: 1)
        }
        if cacheX.dim(1) < 2, isRep {
            cacheX = concatenated([MLXArray.zeros(like: cacheX), cacheX], axis: 1)
        }
        let cacheArg: MLXArray? = isRep ? nil : prev
        let tc = rs.timeConv!(x, cacheX: cacheArg)
        fc[idx] = cacheX
        fi.value += 1
        x = temporalInterleave22(tc, b, t, h, w, c)
    }
    return spatialUpsample22(rs, x)
}

/// One `V22UpResidualBlock` (nested resblocks + Resample + DupUp3D shortcut), cached.
private func upResidualChunk22(
    _ up: V22UpResidualBlock, _ x: MLXArray, _ fc: FeatCache, _ fi: FeatIdx,
    first: Bool, single: Bool
) -> MLXArray {
    let xIn = x
    var xMain = x
    for module in up.upsamples {
        switch module {
        case let r as V22ResidualBlock:
            xMain = r(xMain, featCache: fc, featIdx: fi)
        case let rs as V22Resample:
            if rs.mode == "upsample3d" {
                xMain = resampleUpsample3dCached22(rs, xMain, fc, fi, first: first, single: single)
            } else {  // upsample2d: per-frame, no temporal cache
                xMain = rs(xMain)
            }
        default:
            fatalError("unexpected up-block: \(type(of: module))")
        }
        eval(xMain)
    }
    // DupUp3D shortcut is param-free + per-frame (channel→frame reshape, no causal
    // mixing), so it just needs the first-chunk drop, no cache.
    if let sc = up.avgShortcut {
        let xShort = sc(xIn, firstChunk: first)
        eval(xShort)
        return xMain + xShort
    }
    return xMain
}

/// One latent chunk through V22Decoder3d, threading the shared temporal cache.
/// Returns the head output [B, T', H', W', 12] (pre-unpatchify).
private func decoderChunk22(
    _ decoder: V22Decoder3d, _ x0: MLXArray, _ fc: FeatCache, _ fi: FeatIdx,
    first: Bool, single: Bool
) -> MLXArray {
    var x = convCached22(decoder.conv1, x0, fc, fi)
    for layer in decoder.middle {
        switch layer {
        case let r as V22ResidualBlock: x = r(x, featCache: fc, featIdx: fi)
        case let a as V22AttentionBlock: x = a(x)
        default: fatalError("unexpected middle layer: \(type(of: layer))")
        }
    }
    eval(x)
    for layer in decoder.upsamples {
        x = upResidualChunk22(layer as! V22UpResidualBlock, x, fc, fi, first: first, single: single)
        eval(x)
    }
    x = silu(decoder.head.layer0(x))
    x = convCached22(decoder.head.layer2, x, fc, fi)
    return x
}

/// Lossless temporal-chunked vae22 decode. Bit-identical to `Wan22VAEDecoder(z)` with
/// flat peak memory (≈ one latent chunk's worth). z: DENORMALIZED latent
/// [B, T_lat, H, W, 48] → video [B, T', H', W', 3] in [-1, 1].
public func decodeStreaming22(
    _ decoder: Wan22VAEDecoder, _ z: MLXArray, chunkLat: Int = 1
) -> MLXArray {
    let tLat = z.dim(1)
    let single = tLat == 1  // whole-seq doubles the lone latent frame; a 1-chunk stream must too
    let fc = FeatCache(count: 64)  // generous; one slot per cached conv
    var outs: [MLXArray] = []
    var start = 0
    while start < tLat {
        let first = start == 0
        let zc = z[0..., start..<Swift.min(start + chunkLat, tLat)]
        let xc = decoder.conv2(zc)  // 1×1×1, per-frame, no cache
        let fi = FeatIdx()
        let oc = decoderChunk22(decoder.decoder, xc, fc, fi, first: first, single: single)
        // Materialize oc + the carried cache, else fc holds lazy slice-views into this
        // chunk's freed buffers (alias/go stale past 2 chunks).
        eval([oc] + fc.slots.compactMap { $0 })
        outs.append(oc)
        // Return THIS chunk's transient intermediates (the full-res 1024-ch conv working
        // set — the decode's dominant, spatial-driven allocation) to the OS before the
        // next chunk. Without this the freed-but-cached buffers stack into a high-water
        // ~N_chunks × one-chunk-working-set, which is the decode-internal phys_footprint
        // gap the int4/PAGE_DIT runs isolated. The referenced survivors (oc in `outs`, the
        // carried `fc` slots just eval'd above) are kept; only dead cache is reclaimed, so
        // the result stays bit-identical. Bounds the decode high-water to ~one chunk.
        MLX.Memory.clearCache()
        start += chunkLat
    }
    let video = unpatchify22(concatenated(outs, axis: 1))
    return clip(video, min: MLXArray(Float(-1)), max: MLXArray(Float(1)))
}
