// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Verify",
    platforms: [
        .macOS("27.0"),
    ],
    dependencies: [
        .package(path: "../../CoreAISamplesKit"),
        .package(url: "https://github.com/apple/coreai-models", exact: "1.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.7.0"),
    ],
    targets: [
        .target(
            name: "ControlRuntime",
            dependencies: [
                .product(name: "CoreAILM", package: "coreai-models"),
            ]
        ),
        .executableTarget(
            name: "coreai-verify",
            dependencies: [
                "ControlRuntime",
                .product(name: "CoreAIBackend", package: "CoreAISamplesKit"),
                .product(name: "CoreAILM", package: "coreai-models"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
    ]
)
