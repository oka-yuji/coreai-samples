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
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.5.1"),
    ],
    targets: [
        .target(
            name: "CoreAIBackend",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Jinja", package: "swift-jinja"),
            ]
        ),
        .executableTarget(
            name: "coreai-chat",
            dependencies: ["CoreAIBackend"]
        ),
        .testTarget(
            name: "CoreAIBackendTests",
            dependencies: [
                "CoreAIBackend",
                .product(name: "Jinja", package: "swift-jinja"),
            ]
        ),
    ]
)
