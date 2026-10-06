// swift-tools-version: 6.2
import PackageDescription

let approachableConcurrency: [SwiftSetting] = [
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
    .enableUpcomingFeature("InferIsolatedConformances"),
]

let package = Package(
    name: "CoreAISamplesKit",
    platforms: [
        .macOS("27.0"),
        .iOS("27.0"),
    ],
    products: [
        .library(name: "CoreAIBackend", targets: ["CoreAIBackend"]),
        .library(name: "GLiNERCore", targets: ["GLiNERCore"]),
        .library(name: "GLiNERRuntime", targets: ["GLiNERRuntime"]),
        .library(name: "RAGContext", targets: ["RAGContext"]),
        .executable(name: "coreai-chat", targets: ["coreai-chat"]),
        .library(name: "FeatureStore", targets: ["FeatureStore"]),
        .library(name: "ExtractFeature", targets: ["ExtractFeature"]),
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
        .target(name: "RAGContext"),
        .target(name: "FeatureStore", swiftSettings: approachableConcurrency),
        .target(name: "GLiNERCore", swiftSettings: approachableConcurrency),
        .target(
            name: "ExtractFeature",
            dependencies: ["FeatureStore", "GLiNERCore"],
            swiftSettings: approachableConcurrency
        ),
        .target(
            name: "GLiNERRuntime",
            dependencies: [
                "GLiNERCore",
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: approachableConcurrency
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
        .testTarget(name: "RAGContextTests", dependencies: ["RAGContext"]),
        .testTarget(name: "FeatureStoreTests", dependencies: ["FeatureStore"], swiftSettings: approachableConcurrency),
        .testTarget(
            name: "GLiNERCoreTests",
            dependencies: ["GLiNERCore"],
            resources: [.copy("Resources")],
            swiftSettings: approachableConcurrency
        ),
        .testTarget(
            name: "ExtractFeatureTests",
            dependencies: ["ExtractFeature", "FeatureStore", "GLiNERCore"],
            resources: [.copy("Resources")],
            swiftSettings: approachableConcurrency
        ),
        .testTarget(
            name: "GLiNERRuntimeTests",
            dependencies: ["GLiNERCore", "GLiNERRuntime"],
            swiftSettings: approachableConcurrency
        ),
    ]
)
