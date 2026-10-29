// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "PNGInputKit",
    products: [.library(name: "CImagePNG", targets: ["CImagePNG"])],
    targets: [
        .target(
            name: "CImagePNG",
            cSettings: [
                .define("LODEPNG_NO_COMPILE_ENCODER"),
                .define("LODEPNG_NO_COMPILE_DISK"),
                .define("LODEPNG_NO_COMPILE_ANCILLARY_CHUNKS"),
                .define("LODEPNG_NO_COMPILE_CPP"),
            ])
    ])
