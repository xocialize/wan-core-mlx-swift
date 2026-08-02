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
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        // HV2 weight streaming substrate (BlockStreamKit v0.2.0) — the model-agnostic
        // core extracted from this package's in-tree implementation (b45b879) and
        // ltx-2-mlx-swift's; sibling path dep until the kit repo goes public+tagged.
        .package(path: "../mlx-block-stream-swift"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
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
