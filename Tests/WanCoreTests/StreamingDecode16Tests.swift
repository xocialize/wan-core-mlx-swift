import Foundation
import MLX
import MLXNN
import XCTest

@testable import WanCore

/// `decodeStreaming` (16-ch WanVAE) must be BIT-IDENTICAL to whole-seq `WanVAE.decode` — it only
/// changes WHEN intermediates materialize (per-chunk eval), not WHAT they are. Structural property
/// of the chunking + featCache, independent of weights, so a random-init VAE on a multi-frame latent
/// proves it with no fixtures. 3 latent frames → chunkLat=1 runs 3 chunks (the first-chunk frame-0
/// skip + the >2-chunk aliasing case). CPU stream (deterministic). Gates the VACE decode swap.
final class StreamingDecode16Tests: XCTestCase {
    func testStreamingDecodeMatchesWholeSeq() throws {
        try Device.withDefaultDevice(Device(.cpu)) {
            let vae = WanVAE(zDim: 16, encoder: false)  // decoder path
            eval(vae.parameters())

            // Normalized latent [B, zDim, T_lat, H, W]; 3 latent frames → 3 decode chunks.
            let n = 1 * 16 * 3 * 8 * 8
            let z = MLXArray((0..<n).map { Float(sin(Double($0) * 0.017)) }, [1, 16, 3, 8, 8])
            eval(z)

            let whole = vae.decode(z)
            let streamed = decodeStreaming(vae: vae, z, chunkLat: 1)
            eval(whole, streamed)

            XCTAssertEqual(whole.shape, streamed.shape, "streaming decode shape mismatch")
            let mx = whole.max().item(Float.self), mn = whole.min().item(Float.self)
            XCTAssertTrue(mx.isFinite && mn.isFinite, "whole-seq decode produced non-finite")
            let maxd = abs(whole - streamed).max().item(Float.self)
            print("[WanVAE16 streaming decode] vs whole-seq: max-abs=\(maxd) shape=\(streamed.shape)")
            XCTAssertLessThan(maxd, 1e-4, "streaming decode max-abs \(maxd)")
        }
    }

    /// W5 decisive (a)/(b) split: does `decodeStreaming` PRESERVE temporal order, or REVERSE it?
    /// The existing parity test is order-insensitive (whole-seq would reverse identically). Here we
    /// probe order directly: WanVAE temporal convs are CAUSAL, so an impulse at the LAST latent frame
    /// can only influence the LAST output frames (its causal future). If the multi-chunk FeatCache
    /// carry reverses the sequence, that deviation appears in the FIRST output frames instead.
    /// Needs REAL weights — a random-init WanVAE decodes to all-zeros (conv weights init to 0), which
    /// carries no signal. Loads the VACE 16-ch VAE; skips if /Volumes/DEV_ARCHIVE isn't mounted.
    func testStreamingDecodePreservesTemporalOrder() throws {
        let vaeURL = URL(fileURLWithPath:
            "/Volumes/DEV_ARCHIVE/vace-1.3b-measure/models/vace-1.3b-mlx/vae.safetensors")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: vaeURL.path),
                          "real 16-ch VAE weights not mounted — skipping W5 order probe")
        try Device.withDefaultDevice(Device(.cpu)) {
            let vae = WanVAE(zDim: 16, encoder: true)  // encoder:true loads decoder too
            let loaded = try WeightLoader.loadSafetensors(url: vaeURL)
            WeightLoader.materialize(loaded)
            try vae.update(parameters: ModuleParameters.unflattened(loaded), verify: [.noUnusedKeys])

            // tLat=5 → 17 output frames, exercises the multi-chunk carry.
            let zeros = MLXArray.zeros([1, 16, 5, 8, 8], dtype: .float32)
            // Impulse on the LAST latent frame only (all channels/space): 4 cold frames + 1 hot frame.
            let cold = MLXArray.zeros([1, 16, 4, 8, 8], dtype: .float32)
            let hot = MLXArray.ones([1, 16, 1, 8, 8], dtype: .float32) * Float(4.0)
            let impulse = concatenated([cold, hot], axis: 2)
            eval(zeros, impulse)

            let base = decodeStreaming(vae: vae, zeros, chunkLat: 1)
            let hit = decodeStreaming(vae: vae, impulse, chunkLat: 1)
            eval(base, hit)

            let tOut = hit.dim(2)
            // Per-output-frame L1 deviation introduced by the last-frame impulse.
            var dev = [Float](repeating: 0, count: tOut)
            for t in 0..<tOut {
                dev[t] = abs(hit[0..., 0..., t, 0..., 0...] - base[0..., 0..., t, 0..., 0...])
                    .sum().item(Float.self)
            }
            let half = tOut / 2
            let firstHalf = dev[0..<half].reduce(0, +)
            let secondHalf = dev[half...].reduce(0, +)
            let argmax = dev.firstIndex(of: dev.max()!)!
            print("[W5 order] per-frame dev (\(tOut) frames): \(dev.map { String(format: "%.2f", $0) })")
            print("[W5 order] firstHalf=\(firstHalf) secondHalf=\(secondHalf) argmax=\(argmax)")
            // Last-frame impulse + causal decode ⇒ deviation MUST land in the second half (forward order).
            XCTAssertGreaterThan(
                secondHalf, firstHalf,
                "decodeStreaming REVERSES temporal order: last-frame impulse surfaced in the FIRST half "
                + "(first=\(firstHalf) second=\(secondHalf) argmax=\(argmax)) → W5 bug is in the decode")
        }
    }
}
