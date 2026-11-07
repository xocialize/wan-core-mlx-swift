import Foundation
import MLX
import MLXNN
import XCTest

@testable import WanCore

/// GPU-lane guard for vae22's conv routing (AB-A-0113). mlx's Winograd conv2d (3×3, stride 1,
/// C/O % 32, C+O ≥ 256, N·H·W ≥ 4096) is lossy on M5/NAX GPUs (TF32 batched GEMM, ~6e-3 relL2 per
/// conv in fp32). vae22's per-tap conv2d and Resample conv route those shapes through ExactConv
/// on the GPU. Weight-free: a random V22CausalConv3d and V22Resample at in-window 720p-decoder-like
/// shapes, GPU vs CPU (plain conv2d). The lossy kernel fails this; the exact route passes ~1e-6.
final class VAE22GPUConvGuardTests: XCTestCase {
    private func relL2(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = a - b
        return (sqrt((d * d).sum()) / sqrt((b * b).sum())).item(Float.self)
    }

    func testCausalConv3dGPUMatchesCPUInsideWinogradWindow() throws {
        MLXRandom.seed(5)
        let conv = V22CausalConv3d(256, 256, 3, padding: 1)
        let w = MLXRandom.normal(conv.weight.shape) * (1 / Float(256 * 27).squareRoot())
        let b = MLXRandom.normal(conv.bias.shape) * 0.01
        try conv.update(parameters: ModuleParameters.unflattened(["weight": w, "bias": b]), verify: [.all])
        let x = MLXRandom.normal([1, 2, 64, 64, 256])  // per-frame N·H·W = 4096, C+O = 512
        eval(conv.parameters(), x)
        let gpu = Device.withDefaultDevice(.gpu) { conv(x) }
        let cpu = Device.withDefaultDevice(.cpu) { conv(x) }
        eval(gpu, cpu)
        let r = relL2(gpu, cpu)
        XCTAssertLessThan(r, 1e-4, "vae22 causal conv GPU vs CPU relL2 \(r) — lossy Winograd back on the GPU path?")
    }

    func testResampleUpsampleGPUMatchesCPUInsideWinogradWindow() throws {
        MLXRandom.seed(6)
        let rs = V22Resample(256, mode: "upsample2d")
        let w = MLXRandom.normal(rs.resampleWeight.shape) * (1 / Float(256 * 9).squareRoot())
        try rs.update(parameters: ModuleParameters.unflattened(["resample_weight": w]), verify: [])
        let x = MLXRandom.normal([1, 1, 48, 48, 256])  // upsampled to 96×96 per frame
        eval(rs.parameters(), x)
        let gpu = Device.withDefaultDevice(.gpu) { rs(x) }
        let cpu = Device.withDefaultDevice(.cpu) { rs(x) }
        eval(gpu, cpu)
        let r = relL2(gpu, cpu)
        XCTAssertLessThan(r, 1e-4, "vae22 Resample GPU vs CPU relL2 \(r) — lossy Winograd back on the GPU path?")
    }
}
