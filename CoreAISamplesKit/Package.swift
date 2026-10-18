// swift-tools-version: 6.2
import PackageDescription

let approachableConcurrency: [SwiftSetting] = [
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
    .enableUpcomingFeature("InferIsolatedConformances"),
]

let speechSettings: [SwiftSetting] = [.swiftLanguageMode(.v6)] + approachableConcurrency

let package = Package(
    name: "CoreAISamplesKit",
    defaultLocalization: "en",
    platforms: [
        .macOS("27.0"),
        .iOS("27.0"),
    ],
    products: [
        .library(name: "ModelLibraryUI", targets: ["ModelLibraryUI"]),
        .library(name: "ModelLibraryFeature", targets: ["ModelLibraryFeature"]),
        .library(name: "ObjectFinderFeature", targets: ["ObjectFinderFeature"]),
        .library(name: "ObjectFinderCore", targets: ["ObjectFinderCore"]),
        .library(name: "ObjectFinderRuntime", targets: ["ObjectFinderRuntime"]),
        .executable(name: "object-finder-check", targets: ["object-finder-check"]),
        .library(name: "SpeechCore", targets: ["SpeechCore"]),
        .library(name: "SpeechFeature", targets: ["SpeechFeature"]),
        .library(name: "SpeechRuntime", targets: ["SpeechRuntime"]),
        .library(name: "TruckInspectionFeature", targets: ["TruckInspectionFeature"]),
        .library(name: "TruckInspectionCore", targets: ["TruckInspectionCore"]),
        .library(name: "TruckInspectionRuntime", targets: ["TruckInspectionRuntime"]),
        .library(name: "ImageSearchFeature", targets: ["ImageSearchFeature"]),
        .library(name: "ImageSearchRuntime", targets: ["ImageSearchRuntime"]),
        .library(name: "EmbeddingCore", targets: ["EmbeddingCore"]),
        .library(name: "EmbeddingRuntime", targets: ["EmbeddingRuntime"]),
        .executable(name: "embedding-verify", targets: ["embedding-verify"]),
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
        .library(name: "ConversationFeature", targets: ["ConversationFeature"]),
        .library(name: "ConversationRuntime", targets: ["ConversationRuntime"]),
    ],
    dependencies: [
        .package(path: "../Vendor/IrodoriInputKit"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.4"),
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.5.1"),
    ],
    targets: [
        .target(
            name: "ModelLibraryUI", dependencies: ["ModelLibraryFeature", "FeatureStore"],
            resources: [.process("Resources")], swiftSettings: speechSettings + [.defaultIsolation(MainActor.self)]),
        .target(name: "ModelLibraryFeature", dependencies: ["FeatureStore"], swiftSettings: speechSettings),
        .testTarget(
            name: "ModelLibraryTests", dependencies: ["ModelLibraryFeature", "ModelLibraryUI", "FeatureStore"],
            swiftSettings: speechSettings),
        .target(
            name: "ObjectFinderFeature", dependencies: ["ObjectFinderCore", "FeatureStore"],
            swiftSettings: speechSettings),
        .target(name: "ObjectFinderCore", swiftSettings: speechSettings),
        .target(
            name: "ObjectFinderRuntime",
            dependencies: ["ObjectFinderCore", "ObjectFinderFeature", "EmbeddingCore", "EmbeddingRuntime"],
            swiftSettings: speechSettings),
        .executableTarget(
            name: "object-finder-check", dependencies: ["ObjectFinderCore", "ObjectFinderRuntime"],
            swiftSettings: speechSettings),
        .testTarget(
            name: "ObjectFinderTests",
            dependencies: [
                "ObjectFinderCore", "ObjectFinderFeature", "ObjectFinderRuntime", "FeatureStore", "EmbeddingCore",
                "EmbeddingRuntime",
            ],
            swiftSettings: speechSettings),
        .target(name: "SpeechCore", swiftSettings: speechSettings),
        .target(name: "SpeechFeature", dependencies: ["SpeechCore", "FeatureStore"], swiftSettings: speechSettings),
        .target(
            name: "SpeechRuntime",
            dependencies: [
                "SpeechCore", "SpeechFeature",
                .product(
                    name: "IrodoriInputLogic", package: "IrodoriInputKit", condition: .when(platforms: [.macOS, .iOS])),
                .product(
                    name: "IrodoriTokenizers", package: "IrodoriInputKit", condition: .when(platforms: [.macOS, .iOS])),
            ], resources: [.process("Resources")], swiftSettings: speechSettings),
        .testTarget(name: "SpeechCoreTests", dependencies: ["SpeechCore"], swiftSettings: speechSettings),
        .testTarget(
            name: "SpeechFeatureTests", dependencies: ["SpeechCore", "SpeechFeature", "FeatureStore"],
            swiftSettings: speechSettings),
        .testTarget(
            name: "SpeechRuntimeTests",
            dependencies: [
                "SpeechRuntime", "SpeechCore", "SpeechFeature", "FeatureStore",
                .product(name: "IrodoriInputLogic", package: "IrodoriInputKit"),
                .product(name: "IrodoriTokenizers", package: "IrodoriInputKit"),
            ], resources: [.copy("Resources")], swiftSettings: speechSettings),
        .target(
            name: "TruckInspectionFeature", dependencies: ["TruckInspectionCore", "FeatureStore"],
            swiftSettings: approachableConcurrency),
        .testTarget(
            name: "TruckInspectionFeatureTests",
            dependencies: ["TruckInspectionCore", "TruckInspectionFeature", "TruckInspectionRuntime", "FeatureStore"],
            resources: [.process("Resources")], swiftSettings: approachableConcurrency),
        .target(name: "TruckInspectionCore", swiftSettings: approachableConcurrency),
        .target(
            name: "TruckInspectionRuntime",
            dependencies: ["TruckInspectionCore", "TruckInspectionFeature", "GLiNERRuntime"],
            resources: [.process("Resources")],
            swiftSettings: approachableConcurrency),
        .testTarget(
            name: "TruckInspectionCoreTests", dependencies: ["TruckInspectionCore", "TruckInspectionRuntime"],
            resources: [.process("Resources")], swiftSettings: approachableConcurrency),
        .target(name: "ImageSearchFeature", dependencies: ["FeatureStore"], swiftSettings: approachableConcurrency),
        .target(
            name: "ImageSearchRuntime", dependencies: ["ImageSearchFeature", "EmbeddingCore", "EmbeddingRuntime"],
            swiftSettings: approachableConcurrency),
        .testTarget(
            name: "ImageSearchTests", dependencies: ["ImageSearchFeature", "ImageSearchRuntime", "FeatureStore"],
            swiftSettings: approachableConcurrency),
        .target(name: "EmbeddingCore", swiftSettings: approachableConcurrency),
        .target(
            name: "EmbeddingRuntime",
            dependencies: [
                "EmbeddingCore", .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ], swiftSettings: approachableConcurrency),
        .executableTarget(
            name: "embedding-verify", dependencies: ["EmbeddingCore", "EmbeddingRuntime"],
            swiftSettings: approachableConcurrency),
        .testTarget(
            name: "EmbeddingCoreTests", dependencies: ["EmbeddingCore"], swiftSettings: approachableConcurrency),
        .testTarget(
            name: "EmbeddingRuntimeTests", dependencies: ["EmbeddingCore", "EmbeddingRuntime"],
            resources: [.copy("Fixtures")], swiftSettings: approachableConcurrency),
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
        .target(name: "ConversationFeature", dependencies: ["FeatureStore"], swiftSettings: approachableConcurrency),
        .target(
            name: "ConversationRuntime", dependencies: ["ConversationFeature"],
            swiftSettings: [.swiftLanguageMode(.v6)] + approachableConcurrency),
        .testTarget(
            name: "ConversationRuntimeTests",
            dependencies: ["ConversationRuntime", "ConversationFeature", "FeatureStore"],
            swiftSettings: [.swiftLanguageMode(.v6)] + approachableConcurrency),
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
            name: "ConversationFeatureTests", dependencies: ["ConversationFeature", "FeatureStore"],
            swiftSettings: approachableConcurrency
        ),
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
