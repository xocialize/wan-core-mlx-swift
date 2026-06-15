import Foundation
import MLX
import MLXNN
import XCTest

@testable import WanCore

/// `encodeStreaming` must be BIT-IDENTICAL to the whole-sequence `WanVAE.encode` — it only
/// changes WHEN intermediates materialize (per-chunk eval), not WHAT they are. Bit-identity is
/// a structural property of the chunking + featCache, independent of weight values, so a
/// fixed-seed random-init VAE on a multi-chunk input proves it with no fixtures. T=9 → 3 chunks
/// (frame 0, 1–4, 5–8) exercises the >2-chunk featCache aliasing case (the per-chunk slot-eval
/// staleness fix). Runs on the CPU stream (deterministic).
final class StreamingEncodeTests: XCTestCase {
    func testStreamingEncodeMatchesWholeSeq() throws {
        try Device.withDefaultDevice(Device(.cpu)) {
            // Same VAE instance feeds both paths, so bit-identity holds for ANY weights — no
            // seeding needed. Deterministic, varied input (no MLXRandom dep in this test target).
            let vae = WanVAE(zDim: 16, encoder: true)
            eval(vae.parameters())

            let n = 1 * 3 * 9 * 32 * 32
            let x = MLXArray((0..<n).map { Float(sin(Double($0) * 0.013)) }, [1, 3, 9, 32, 32])
            eval(x)

            let whole = vae.encode(x)
            let streamed = encodeStreaming(vae: vae, x)
            eval(whole, streamed)

            XCTAssertEqual(whole.shape, streamed.shape, "streaming encode shape mismatch")
            let mx = whole.max().item(Float.self), mn = whole.min().item(Float.self)
            XCTAssertTrue(mx.isFinite && mn.isFinite, "whole-seq encode produced non-finite")
            let maxd = abs(whole - streamed).max().item(Float.self)
            print("[WanVAE streaming encode] vs whole-seq: max-abs=\(maxd) shape=\(streamed.shape)")
            XCTAssertLessThan(maxd, 1e-4, "streaming encode max-abs \(maxd)")
        }
    }
}
