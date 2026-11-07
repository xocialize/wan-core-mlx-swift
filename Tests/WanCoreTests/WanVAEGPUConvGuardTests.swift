import Foundation
import MLX
import MLXNN
import XCTest

@testable import WanCore

/// GPU-lane guard for the 16-ch WanVAE's conv routing (AB-A-0113). mlx's Winograd conv2d is lossy
/// on M5/NAX GPUs (TF32, ~6e-3 relL2 per conv in fp32), and mlx-swift 0.32.x splits a 3×3×3 conv3d
/// into per-frame Winograd conv2d. On the GPU, CausalConv3d and the Resample upsample conv go
/// through ExactConv. Weight-free: random modules at in-window 480p-decoder-like shapes, GPU vs CPU
/// (the CPU path is unchanged). Fails under WAN_VAE_CONV_ROUTE=raw.
final class WanVAEGPUConvGuardTests: XCTestCase {
    private func relL2(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = a - b
        return (sqrt((d * d).sum()) / sqrt((b * b).sum())).item(Float.self)
    }

    func testCausalConv3dGPUMatchesCPUInsideWinogradWindow() throws {
        MLXRandom.seed(7)
        let conv = CausalConv3d(192, 192, 3, padding: 1)
        let w = MLXRandom.normal(conv.weight.shape) * (1 / Float(192 * 27).squareRoot())
        let b = MLXRandom.normal(conv.bias.shape) * 0.01
        try conv.update(parameters: ModuleParameters.unflattened(["weight": w, "bias": b]), verify: [.all])
        let x = MLXRandom.normal([1, 192, 3, 64, 64])  // channel-first; per-frame 4096 px, C+O = 384
        eval(conv.parameters(), x)
        let gpu = Device.withDefaultDevice(.gpu) { conv(x) }
        let cpu = Device.withDefaultDevice(.cpu) { conv(x) }
        eval(gpu, cpu)
        let r = relL2(gpu, cpu)
        XCTAssertLessThan(r, 1e-4, "WanVAE CausalConv3d GPU vs CPU relL2 \(r) — lossy Winograd back on the GPU path?")
    }

    func testResampleUpsampleConvGPUMatchesCPUInsideWinogradWindow() throws {
        MLXRandom.seed(8)
        let rs = Resample(384, mode: "upsample2d")
        let p = rs.parameters().flattened()
        var upd: [String: MLXArray] = [:]
        for (k, v) in p { upd[k] = MLXRandom.normal(v.shape) * (k.hasSuffix("weight") ? 1 / Float(384 * 9).squareRoot() : 0.01) }
        try rs.update(parameters: ModuleParameters.unflattened(upd), verify: [.all])
        let x = MLXRandom.normal([96, 96, 384]).reshaped(1, 96, 96, 384)  // NHWC, upsampled input
        eval(rs.parameters(), x)
        let gpu = Device.withDefaultDevice(.gpu) { rs.upsampleConv(x) }
        let cpu = Device.withDefaultDevice(.cpu) { rs.upsampleConv(x) }
        eval(gpu, cpu)
        let r = relL2(gpu, cpu)
        XCTAssertLessThan(r, 1e-4, "WanVAE Resample upsample conv GPU vs CPU relL2 \(r) — lossy Winograd back on the GPU path?")
    }
}
