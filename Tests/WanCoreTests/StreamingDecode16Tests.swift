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
}
