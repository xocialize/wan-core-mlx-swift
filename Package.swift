// swift-tools-version: 6.2
// wan-core-mlx-swift — the neutral Wan substrate (DiT + 16-ch WanVAE + umT5 + RoPE + schedulers +
// strict safetensors loader) extracted from bernini-r-mlx-swift so the Wan family (Bernini-R, Helios,
// Phantom, TI2V-5B) shares ONE MLX-bearing core. Fix once → every consumer inherits. Engine-agnostic
// (no MLXToolKit dep); wrappers live in the per-model packages.

import PackageDescription

let package = Package(
    name: "WanCore",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "WanCore", targets: ["WanCore"]),
        // HV2 weight streaming (NEUROSTREAM-ACTIONS): granule layout tool +
        // the streaming receipts gate (parity / runtime gate / 16 GB emulation).
        .executable(name: "wan-granule-layout", targets: ["GranuleLayoutCLI"]),
        .executable(name: "RunWanStream", targets: ["RunWanStream"]),
    ],
    dependencies: [
        // Floor 0.32.3: <= 0.31.6 JIT-miscompiles the NAX split-K bf16 GEMM (mlx#3797/#3810), which
        // breaks TI2V-5B ffn.fc2 at B·L 1366–4778 (AB-R-0778). Ceiling matches mlx-block-stream-swift.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", "0.32.3"..<"0.33.0"),
        // HV2 weight streaming substrate (BlockStreamKit) — the model-agnostic core
        // extracted from this package's in-tree implementation (b45b879) and
        // ltx-2-mlx-swift's. Versioned dep since 2026-08-14: the kit repo went public +
        // tagged v0.5.0, the condition the sibling path dep was waiting on.
        .package(url: "https://github.com/xocialize/mlx-block-stream-swift", from: "0.5.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // Exact-class GPU convs for vae22's decoder (AB-A-0113): mlx's Winograd conv2d is lossy on NAX.
        .package(url: "https://github.com/xocialize/mlx-exact-conv-swift", from: "0.1.1"),
    ],
    targets: [
        .target(
            name: "WanCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "BlockStreamKit", package: "mlx-block-stream-swift"),
                .product(name: "MLXExactConv", package: "mlx-exact-conv-swift"),
            ],
            path: "Sources/WanCore"
        ),
        .executableTarget(
            name: "GranuleLayoutCLI",
            dependencies: ["WanCore"],
            path: "Sources/GranuleLayoutCLI"
        ),
        .executableTarget(
            name: "RunWanStream",
            dependencies: [
                "WanCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/RunWanStream"
        ),
        .testTarget(
            name: "WanCoreTests",
            dependencies: [
                "WanCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Tests/WanCoreTests"
        ),
    ]
)
