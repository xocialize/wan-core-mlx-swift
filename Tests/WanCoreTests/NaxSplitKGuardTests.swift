import Foundation
import MLX
import MLXNN
import XCTest

@testable import WanCore

/// GPU regression guard for the NAX split-K bf16 GEMM miscompile (ml-explore/mlx#3797, fixed by
/// #3810, first shipped in mlx-swift 0.32.3). On mlx-swift <= 0.31.6 a half-precision GEMM with
/// M·N >= 2048², K >= 10240 and K >= 3·max(M, N) returned garbage/NaN on M5-class GPUs. In wan-core
/// that is exactly TI2V-5B's `ffn.fc2` (K = 14336 -> N = 3072) for B·L in [1366, 4778] rows: the
/// "Metal bf16 is unreliable at large seqLen" finding (AB-R-0776 -> AB-R-0778).
///
/// Weight-free: a random-weight WanFFN at the TI2V-5B shape, bf16 on the GPU vs fp32 on the CPU, at
/// row counts inside the old window. Fails if the package ever resolves a broken mlx-swift again.
final class NaxSplitKGuardTests: XCTestCase {
    func testTI2V5BFFNBf16MatchesFp32InsideOldWindow() throws {
        MLXRandom.seed(3)
        let ffn = WanFFN(dim: 3072, ffnDim: 14336)
        let p32 = ffn.parameters().mapValues { (($0 as MLXArray) * 1).asType(.float32) }
        eval(p32)
        let ffn32 = WanFFN(dim: 3072, ffnDim: 14336)
        try ffn32.update(parameters: p32, verify: [.all])
        ffn.update(parameters: p32.mapValues { $0.asType(.bfloat16) })
        eval(ffn.parameters(), ffn32.parameters())

        for rows in [1366, 2640] {  // both inside the 0.31.x dispatch window
            let x = MLXRandom.normal([1, rows, 3072]).asType(.bfloat16)
            eval(x)
            let gpu = Device.withDefaultDevice(.gpu) { ffn(x).asType(.float32) }
            let cpu = Device.withDefaultDevice(.cpu) { ffn32(x.asType(.float32)) }
            eval(gpu, cpu)
            XCTAssertTrue(gpu.abs().max().item(Float.self).isFinite, "rows=\(rows): NaN/Inf from bf16 fc2")
            let d = gpu - cpu
            let rel = (sqrt((d * d).sum()) / sqrt((cpu * cpu).sum())).item(Float.self)
            // bf16 rounding alone is ~3e-3; the miscompiled kernel gives ~1 (or NaN).
            XCTAssertLessThan(rel, 2e-2, "rows=\(rows): bf16 WanFFN relL2 \(rel) vs fp32 — NAX split-K regression?")
        }
    }
}
