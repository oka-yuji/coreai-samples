// swift-tools-version: 6.2
import Foundation
import PackageDescription

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let settings: [SwiftSetting] = [
    .swiftLanguageMode(.v6),
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
    .enableUpcomingFeature("InferIsolatedConformances"),
]
let package = Package(
    name: "IrodoriInputKit", platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "IrodoriInputLogic", targets: ["IrodoriInputLogic"]),
        .library(name: "IrodoriTokenizers", targets: ["IrodoriTokenizers"]),
    ],
    targets: [
        .target(name: "IrodoriInputLogic", swiftSettings: settings),
        .target(
            name: "CTokenizers", publicHeadersPath: "include",
            linkerSettings: [
                .unsafeFlags(
                    ["-L", root.appendingPathComponent("rust/target/release").path], .when(platforms: [.macOS])),
                .unsafeFlags(
                    ["-L", root.appendingPathComponent("rust/target/aarch64-apple-ios/release").path],
                    .when(platforms: [.iOS])),
                .linkedLibrary("irodori_tokenizers"), .linkedLibrary("c++"),
            ]),
        .target(
            name: "IrodoriTokenizers", dependencies: ["IrodoriInputLogic", "CTokenizers"], swiftSettings: settings),
    ])
