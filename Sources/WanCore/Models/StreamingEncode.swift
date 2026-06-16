// Lossless temporal-chunked ENCODE — the `decodeStreaming` analog (E15 Addendum 8).
// `WanVAE.encode` already chunks the encoder forward temporally, but accumulates every
// chunk into one lazy graph and only materializes at the caller's `eval` — so the full-res
// fp32 conv working set of the WHOLE sequence is live at once (the VACE VCU-build wall:
// ~41 GB live + glacial under a tight cache cap). This version `eval`s each chunk's forward
// + the carried `featCache` BEFORE the next chunk, so the live working set is ONE chunk and
// the cache cap no longer thrashes a multi-chunk graph. The accumulated chunk outputs are at
// the downsampled latent resolution (small), so the final `conv1` + normalization on the whole
// is cheap. Bit-identical to `vae.encode(x)` (eval only materializes; the per-chunk featCache
// materialization is the same staleness fix `decodeStreaming` uses).

import Foundation
import MLX

/// Temporal-chunked encode. x: video [B, 3, T, H, W] in [-1, 1] ->
/// normalized latent [B, zDim, T_lat, H_lat, W_lat]. Bit-identical to `WanVAE.encode`,
/// flat peak memory. Falls back to `encode` for a decode-only VAE (no encoder built).
public func encodeStreaming(vae: WanVAE, _ x: MLXArray) -> MLXArray {
    guard let encoder = vae.encoder, let conv1 = vae.conv1 else {
        return vae.encode(x)  // decode-only VAE — nothing to stream
    }

    let t = x.dim(2)
    let numChunks = 1 + (t - 1) / 4  // frame 0 alone, then 4-frame chunks — matches `encode`
    let fc = FeatCache(count: vae._countEncoderCacheSlots())
    var outs: [MLXArray] = []

    for i in 0..<numChunks {
        let chunk = i == 0
            ? x[0..., 0..., ..<1]
            : x[0..., 0..., (1 + 4 * (i - 1))..<(1 + 4 * i)]
        let fi = FeatIdx()
        let oc = encoder(chunk, featCache: fc, featIdx: fi)
        // Materialize the carried cache too — else `fc` holds lazy slice-views into this
        // chunk's freed buffers, which alias/go stale across the boundary (same fix as decode).
        eval([oc] + fc.slots.compactMap { $0 })
        outs.append(oc)
        // Per-chunk progress (E15 Addendum 10): proves chunking ENGAGES (chunk i/N progressing
        // = linear, NOT a loop) and shows the per-chunk active/cache cost. `WAN_VAE_LOG=1`.
        if ProcessInfo.processInfo.environment["WAN_VAE_LOG"] != nil {
            let s = Memory.snapshot()
            func gb(_ b: Int) -> String { String(format: "%.1f", Double(b) / 1e9) }
            print("[WanVAE encode] chunk \(i + 1)/\(numChunks) (frames \(chunk.dim(2))): "
                + "active=\(gb(s.activeMemory)) cache=\(gb(s.cacheMemory)) peak=\(gb(s.peakMemory)) GB")
        }
    }

    let out = concatenated(outs, axis: 2)
    let mu = split(conv1(out), parts: 2, axis: 1)[0]
    let mean = vae.mean.reshaped(1, -1, 1, 1, 1)
    let invStd = vae.invStd.reshaped(1, -1, 1, 1, 1)
    return (mu - mean) * invStd
}
