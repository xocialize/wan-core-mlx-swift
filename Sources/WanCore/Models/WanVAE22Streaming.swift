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
// cache) and the top-level chunk loop — are filled here. Phase-1 bounds the TEMPORAL
// extent; Phase-2 (below) optionally spatial-halo-tiles the high-res suffix to bound the
// per-frame spatial peak too (exact: real-neighbour halo + crop, no blend).

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

/// Index of the first `V22UpResidualBlock` with NO `upsample3d` (temporal) resample —
/// the start of the spatially-tileable suffix. Earlier blocks carry the cross-chunk
/// temporal-upsample `Rep` state and the only spatially-global op (the middle attention
/// is upstream of all upsamples), so they must run WHOLE-spatial. (Ports vae_stream.py
/// `_suffix_start`; the suffix = the high-res tail where the decode's peak lives.)
private func suffixStart22(_ decoder: V22Decoder3d) -> Int {
    for (i, layer) in decoder.upsamples.enumerated() {
        let up = layer as! V22UpResidualBlock
        let hasTemporal = up.upsamples.contains { ($0 as? V22Resample)?.mode == "upsample3d" }
        if !hasTemporal { return i }
    }
    return decoder.upsamples.count
}

/// conv1 → middle → upsamples[..<suffixStart], WHOLE spatial. Threads the shared
/// temporal cache exactly as the single-pass path; returns the suffix input. (Ports
/// vae_stream.py `_decoder3d_prefix`.)
private func decoderPrefix22(
    _ decoder: V22Decoder3d, _ x0: MLXArray, _ fc: FeatCache, _ fi: FeatIdx,
    first: Bool, single: Bool, suffixStart: Int
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
    for layer in decoder.upsamples[..<suffixStart] {
        x = upResidualChunk22(layer as! V22UpResidualBlock, x, fc, fi, first: first, single: single)
        eval(x)
    }
    return x
}

/// upsamples[suffixStart...] → head, on a (possibly spatially-sliced, in Phase 2) input.
/// No `upsample3d` here, so temporal causality is carried entirely by the CausalConv3d
/// caches in `fc`; the suffix convs are spatially-LOCAL (finite receptive field) — that
/// locality is exactly what makes the Phase-2 halo-tile of this segment bit-identical.
/// (Ports vae_stream.py `_decoder3d_suffix`.)
private func decoderSuffix22(
    _ decoder: V22Decoder3d, _ x0: MLXArray, _ fc: FeatCache, _ fi: FeatIdx,
    first: Bool, single: Bool, suffixStart: Int
) -> MLXArray {
    var x = x0
    for layer in decoder.upsamples[suffixStart...] {
        x = upResidualChunk22(layer as! V22UpResidualBlock, x, fc, fi, first: first, single: single)
        eval(x)
    }
    x = silu(decoder.head.layer0(x))
    x = convCached22(decoder.head.layer2, x, fc, fi)
    return x
}

/// One latent chunk through V22Decoder3d, threading the shared temporal cache.
/// Returns the head output [B, T', H', W', 12] (pre-unpatchify). Split into a
/// whole-spatial prefix + a (Phase-2-tileable) suffix sharing one continuous
/// `featIdx`/`featCache` → byte-for-byte the original single-pass behaviour.
private func decoderChunk22(
    _ decoder: V22Decoder3d, _ x0: MLXArray, _ fc: FeatCache, _ fi: FeatIdx,
    first: Bool, single: Bool
) -> MLXArray {
    let suffixStart = suffixStart22(decoder)
    let suffixIn = decoderPrefix22(decoder, x0, fc, fi, first: first, single: single, suffixStart: suffixStart)
    return decoderSuffix22(decoder, suffixIn, fc, fi, first: first, single: single, suffixStart: suffixStart)
}

// ---------------------------------------------------------------------------
// Phase 2: spatial halo-tile + CROP of the high-res suffix.
//
// Phase-1 caps the TEMPORAL extent (one-chunk floor). The remaining peak lives in
// the high-res suffix (the full-res upsample tail). We split the decoder at the first
// non-temporal upsample block (`suffixStart22`): the PREFIX (incl. the only spatially-
// global op, the middle attention) runs WHOLE at low res; the SUFFIX is spatially LOCAL
// (finite conv receptive field), so it can be tiled. Each tile pulls a real-neighbour
// halo (≥ the suffix RF) from the full prefix output, runs the suffix unchanged, then
// CROPS the halo away — no blend, no fade. A kept pixel ≥halo from the slice edge sees
// only the input whole-seq would have given it ⇒ bit-identical. Tiles compose with the
// temporal streaming via a per-tile temporal cache (the suffix convs are temporally-
// causal AND spatially-local; the halo bounds the spatial reach inside the cached
// boundary frames too). Ports vae_stream.py's Phase-2 block.
// ---------------------------------------------------------------------------

/// Suffix receptive-field halo in suffix-input (G) pixels. The vae22 suffix RF is 10
/// (verified in the Python reference three ways incl. a halo-sweep cliff at 10); +2
/// margin. The bit-identity gate (max|Δ|=0), not an effective-RF threshold, is the real
/// guarantee. (Ports vae_stream.py `_SUFFIX_HALO`.)
let suffixHalo22 = 12

/// Partition [0, size) into n contiguous, near-equal tiles (drops empties).
private func tileBounds22(_ size: Int, _ n: Int) -> [(Int, Int)] {
    let base = size / n, rem = size % n
    var bounds: [(Int, Int)] = []
    var s = 0
    for i in 0..<n {
        let e = s + base + (i < rem ? 1 : 0)
        if e > s { bounds.append((s, e)) }
        s = e
    }
    return bounds
}

/// The spatial ×-factor the suffix applies (product of its resamples; each `V22Resample`
/// doubles H and W). Used to map suffix-INPUT tile/halo bounds → suffix-OUTPUT crop bounds.
private func suffixSpatialScale22(_ decoder: V22Decoder3d, suffixStart: Int) -> Int {
    var scale = 1
    for layer in decoder.upsamples[suffixStart...] {
        let up = layer as! V22UpResidualBlock
        for module in up.upsamples where module is V22Resample { scale *= 2 }
    }
    return scale
}

/// No silent cap: log when the halo makes every tile span the whole axis so spatial
/// tiling saves no memory at this resolution (still correct, just a no-op).
private func warnIfIneffective22(_ gH: Int, _ gW: Int, _ n: Int, _ halo: Int) {
    let rb = tileBounds22(gH, n), cb = tileBounds22(gW, n)
    let maxH = rb.map { Swift.min(gH, $0.1 + halo) - Swift.max(0, $0.0 - halo) }.max() ?? gH
    let maxW = cb.map { Swift.min(gW, $0.1 + halo) - Swift.max(0, $0.0 - halo) }.max() ?? gW
    if maxH >= gH && maxW >= gW {
        FileHandle.standardError.write(Data(
            "[WanVAE22] spatial_tiles=\(n) at suffix grid \(gH)x\(gW): halo=\(halo) makes the largest tile span the whole grid → tiling saves no memory at this resolution (output still correct, a no-op).\n".utf8))
    }
}

/// Run the suffix on x=(B,T,G_h,G_w,C) spatially tiled n×n with a real-neighbour halo +
/// crop → bit-identical to the whole suffix, but with peak ≈ one tile's working set.
/// `fcTiles[k]` is tile k's persistent temporal cache (same spatial bounds every chunk).
/// `keepCache == false` (single-chunk decode): no successor reads the per-tile caches, so
/// drop each right after its tile → footprint flat in tile-count. (Ports vae_stream.py
/// `_suffix_spatial_tiled`.)
private func suffixSpatialTiled22(
    _ decoder: V22Decoder3d, _ x: MLXArray, _ fcTiles: inout [FeatCache?],
    suffixStart: Int, nTiles: Int, halo: Int, scale: Int, keepCache: Bool
) -> MLXArray {
    let gH = x.dim(2), gW = x.dim(3)
    let rb = tileBounds22(gH, nTiles), cb = tileBounds22(gW, nTiles)
    var rowStrips: [MLXArray] = []
    var k = 0
    for (r0, r1) in rb {
        var colPieces: [MLXArray] = []
        for (c0, c1) in cb {
            // pull a real-neighbour halo from the FULL prefix output; clamp at the global
            // edge (there the conv's own zero-pad matches whole-seq exactly).
            let sr0 = Swift.max(0, r0 - halo), sr1 = Swift.min(gH, r1 + halo)
            let sc0 = Swift.max(0, c0 - halo), sc1 = Swift.min(gW, c1 + halo)
            let sl = x[0..., 0..., sr0..<sr1, sc0..<sc1, 0...]
            let fi = FeatIdx()
            let tileCache = fcTiles[k] ?? FeatCache(count: 64)
            var o = decoderSuffix22(decoder, sl, tileCache, fi, first: false, single: false, suffixStart: suffixStart)
            // crop the halo: input slice began at sr0 ⇒ output (×scale) began at scale*sr0.
            let cr0 = scale * (r0 - sr0), cc0 = scale * (c0 - sc0)
            o = o[0..., 0..., cr0..<(cr0 + scale * (r1 - r0)), cc0..<(cc0 + scale * (c1 - c0)), 0...]
            if keepCache {
                eval([o] + tileCache.slots.compactMap { $0 })  // carry this tile's cache to the next chunk
                fcTiles[k] = tileCache
            } else {
                fcTiles[k] = nil                                // single chunk: cache never read → free it
                eval(o)
            }
            colPieces.append(o)
            k += 1
            MLX.Memory.clearCache()
        }
        let strip = colPieces.count == 1 ? colPieces[0] : concatenated(colPieces, axis: 3)
        eval(strip)
        rowStrips.append(strip)
    }
    return rowStrips.count == 1 ? rowStrips[0] : concatenated(rowStrips, axis: 2)
}

/// Pick `spatialTiles` for `decodeStreaming22` from the latent H/W. The tileable suffix
/// runs on grid G = (suffix-input scale)·lat; aim for ~`targetGPx` G-pixels per tile.
/// Returns 1 for small images (tiling would be a no-op). `maxN` caps the result — pass it
/// for VIDEO (multi-chunk keeps the per-tile caches live, ~U-curve in n); images leave it
/// nil (single-chunk drops caches → higher n is free). (Ports vae_stream.py
/// `suggest_spatial_tiles`; here G scales with the prefix output, default factor 4·lat.)
public func suggestSpatialTiles22(hLat: Int, wLat: Int, targetGPx: Int = 16, maxN: Int? = nil) -> Int {
    let g = 4 * Swift.max(hLat, wLat)
    var n = Swift.max(1, Int((Double(g) / Double(targetGPx)).rounded()))
    if let maxN { n = Swift.min(n, maxN) }
    return n
}

/// Lossless streaming vae22 decode — temporal (Phase 1) + optional spatial halo-tiling of
/// the high-res suffix (Phase 2). Bit-identical to `Wan22VAEDecoder(z)` with flat peak
/// memory. z: DENORMALIZED latent [B, T_lat, H, W, 48] → video [B, T', H', W', 3] in [-1, 1].
///   - chunkLat: latent frames per temporal chunk (1 = lowest memory). Result-invariant.
///   - spatialTiles: n → tile the high-res suffix n×n (real-neighbour halo + crop). 1 =
///     Phase-1 behaviour. Result-invariant (given halo ≥ suffix RF).
///   - halo: suffix-input halo in G-pixels (≥ suffix RF = 10). nil → `suffixHalo22` (12).
public func decodeStreaming22(
    _ decoder: Wan22VAEDecoder, _ z: MLXArray, chunkLat: Int = 1,
    spatialTiles: Int = 1, halo: Int? = nil
) -> MLXArray {
    let tLat = z.dim(1)
    let single = tLat == 1  // whole-seq doubles the lone latent frame; a 1-chunk stream must too
    let h = halo ?? suffixHalo22
    let suffixStart = suffixStart22(decoder.decoder)
    let scale = suffixSpatialScale22(decoder.decoder, suffixStart: suffixStart)
    // multi-chunk (video) keeps the per-tile suffix caches live; one chunk (image) drops them.
    let keepCache = chunkLat < tLat
    let fc = FeatCache(count: 64)  // generous; one slot per cached conv (prefix temporal cache)
    var fcTiles: [FeatCache?] = []  // lazily sized once the suffix-grid G is known (tiled path)
    var outs: [MLXArray] = []
    var start = 0
    while start < tLat {
        let first = start == 0
        let zc = z[0..., start..<Swift.min(start + chunkLat, tLat)]
        let xc = decoder.conv2(zc)  // 1×1×1, per-frame, no cache
        let fi = FeatIdx()
        let oc: MLXArray
        if spatialTiles <= 1 {
            oc = decoderChunk22(decoder.decoder, xc, fc, fi, first: first, single: single)
        } else {
            let suffixIn = decoderPrefix22(decoder.decoder, xc, fc, fi, first: first, single: single, suffixStart: suffixStart)
            eval(suffixIn)
            if fcTiles.isEmpty {  // one persistent temporal cache per real tile
                let gH = suffixIn.dim(2), gW = suffixIn.dim(3)
                let nReal = tileBounds22(gH, spatialTiles).count * tileBounds22(gW, spatialTiles).count
                fcTiles = Array(repeating: nil, count: nReal)
                warnIfIneffective22(gH, gW, spatialTiles, h)
            }
            oc = suffixSpatialTiled22(decoder.decoder, suffixIn, &fcTiles,
                                      suffixStart: suffixStart, nTiles: spatialTiles, halo: h, scale: scale, keepCache: keepCache)
        }
        // Materialize oc + the carried PREFIX cache, else fc holds lazy slice-views into
        // this chunk's freed buffers (alias/go stale past 2 chunks).
        eval([oc] + fc.slots.compactMap { $0 })
        outs.append(oc)
        // Return THIS chunk's transient intermediates (the full-res conv working set — the
        // decode's dominant, spatial-driven allocation) to the OS before the next chunk, so
        // the high-water stays ~one chunk rather than N_chunks × one-chunk-working-set. The
        // referenced survivors (oc, the carried fc/fcTiles slots just eval'd) are kept; only
        // dead cache is reclaimed, so the result stays bit-identical.
        MLX.Memory.clearCache()
        start += chunkLat
    }
    let video = unpatchify22(concatenated(outs, axis: 1))
    return clip(video, min: MLXArray(Float(-1)), max: MLXArray(Float(1)))
}
