// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "CoreAISamples",
    platforms: [
        .macOS("27.0"),
        .iOS("27.0"),
    ],
    products: [
        .library(name: "CoreAIBackend", targets: ["CoreAIBackend"]),
        .executable(name: "coreai-chat", targets: ["coreai-chat"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.4"),
    ],
    targets: [
        .target(
            name: "CoreAIBackend",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "coreai-chat",
            dependencies: ["CoreAIBackend"]
        ),
        .testTarget(
            name: "CoreAIBackendTests",
            dependencies: ["CoreAIBackend"]
        ),
    ]
)
