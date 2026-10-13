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
        .library(name: "ProductFeature", targets: ["ProductFeature"]),
        .library(name: "ProductCore", targets: ["ProductCore"]),
        .library(name: "ProductRuntime", targets: ["ProductRuntime"]),
        .library(name: "GLiNERCore", targets: ["GLiNERCore"]),
        .library(name: "GLiNERRuntime", targets: ["GLiNERRuntime"]),
        .library(name: "RAGContext", targets: ["RAGContext"]),
        .executable(name: "coreai-chat", targets: ["coreai-chat"]),
        .library(name: "FeatureStore", targets: ["FeatureStore"]),
        .library(name: "ExtractFeature", targets: ["ExtractFeature"]),
        .library(name: "DecideFeature", targets: ["DecideFeature"]),
        .library(name: "DecideRuntime", targets: ["DecideRuntime"]),
        .library(name: "RecipeCore", targets: ["RecipeCore"]),
        .library(name: "RecipeRuntime", targets: ["RecipeRuntime"]),
        .library(name: "RecipeFeature", targets: ["RecipeFeature"]),
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
            ],
            resources: [.process("Resources")]
        ),
        .target(name: "RAGContext"),
        .target(
            name: "ProductFeature", dependencies: ["ProductCore", "FeatureStore"],
            swiftSettings: approachableConcurrency),
        .testTarget(
            name: "ProductFeatureTests", dependencies: ["ProductFeature", "ProductCore", "ProductRuntime"],
            swiftSettings: approachableConcurrency),
        .target(name: "ProductCore", swiftSettings: approachableConcurrency),
        .target(
            name: "ProductRuntime", dependencies: ["ProductCore", "ProductFeature", "GLiNERCore", "GLiNERRuntime"],
            resources: [.process("Resources")], swiftSettings: approachableConcurrency),
        .testTarget(name: "ProductCoreTests", dependencies: ["ProductCore"], swiftSettings: approachableConcurrency),
        .testTarget(
            name: "ProductRuntimeTests", dependencies: ["ProductRuntime", "ProductCore"],
            swiftSettings: approachableConcurrency),
        .target(name: "RecipeCore", swiftSettings: approachableConcurrency),
        .target(
            name: "RecipeFeature", dependencies: ["FeatureStore", "RecipeCore"], swiftSettings: approachableConcurrency),
        .target(
            name: "RecipeRuntime", dependencies: ["RecipeCore", "RecipeFeature", "CoreAIBackend"],
            resources: [.process("Resources")],
            swiftSettings: approachableConcurrency
        ),
        .target(name: "DecideFeature", dependencies: ["FeatureStore"], swiftSettings: approachableConcurrency),
        .target(
            name: "DecideRuntime", dependencies: ["DecideFeature", "CoreAIBackend"],
            swiftSettings: approachableConcurrency
        ),
        .target(name: "FeatureStore", swiftSettings: approachableConcurrency),
        .target(name: "GLiNERCore", resources: [.process("Resources")], swiftSettings: approachableConcurrency),
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
            ],
            resources: [.copy("Fixtures")]
        ),
        .testTarget(name: "RAGContextTests", dependencies: ["RAGContext"]),
        .testTarget(name: "RecipeCoreTests", dependencies: ["RecipeCore"], swiftSettings: approachableConcurrency),
        .testTarget(
            name: "RecipeFeatureTests", dependencies: ["RecipeFeature", "RecipeCore", "FeatureStore"],
            swiftSettings: approachableConcurrency),
        .testTarget(
            name: "RecipeRuntimeTests", dependencies: ["RecipeRuntime", "RecipeCore", "RecipeFeature", "FeatureStore"],
            resources: [.process("Resources")],
            swiftSettings: approachableConcurrency
        ),
        .testTarget(
            name: "DecideFeatureTests", dependencies: ["DecideFeature", "FeatureStore"],
            swiftSettings: approachableConcurrency
        ),
        .testTarget(
            name: "DecideRuntimeTests", dependencies: ["DecideRuntime", "DecideFeature", "FeatureStore"],
            swiftSettings: approachableConcurrency
        ),
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
