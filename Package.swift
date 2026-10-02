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
    ],
    targets: [
        .target(name: "CoreAIBackend"),
    ]
)
