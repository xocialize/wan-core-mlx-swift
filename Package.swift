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
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
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
            ],
            path: "Sources/WanCore"
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
