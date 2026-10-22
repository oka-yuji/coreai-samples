import CryptoKit
import Foundation
import ImageGenerationCore
import ImageGenerationRuntime
import Testing

struct ImageGenerationImageTests {
    @Test("Straight PNG channels EXIF orientation and the tagged RGB image retain the original tensor pixels")
    func readerAndOrientationMatchPillow() throws {
        let golden = try ImageGolden.load()
        let source = try ImageGenerationImageReader.read(NativeGolden.resource("rgba-source.png"))
        #expect(hash(source.rgba) == golden.sourceSHA256)
        let tagged = try ImageGenerationImageReader.read(NativeGolden.resource("rgba-srgb.png"))
        #expect(hash(tagged.rgba) == golden.sourceSHA256)
        let oriented = try ImageGenerationImageReader.read(NativeGolden.resource("rgba-oriented.png"))
        #expect(hash(oriented.rgba) == golden.orientedSHA256)
        for sample in golden.orientations {
            let result = try source.oriented(#require(sample.orientation))
            #expect(result.width == sample.width && result.height == sample.height)
            #expect(hash(result.rgba) == sample.rgbaSHA256)
        }
    }

    @Test(
        "RGBA resizing matches Pillow premultiplication filtering and unpremultiplication including transparent pixels")
    func resizingMatchesPillow() throws {
        let golden = try ImageGolden.load()
        let source = try ImageGenerationImageReader.read(NativeGolden.resource("rgba-source.png"))
        for sample in golden.resizes {
            let resized = try source.resized(width: sample.width, height: sample.height)
            #expect(hash(resized.rgba) == sample.rgbaSHA256)
        }
        let canonical = try source.squareReference()
        #expect(canonical.width == 512 && canonical.height == 512)
        #expect(hash(canonical.rgba) == golden.canonicalSHA256)
        #expect(ImageGenerationPixels.referenceVersion == golden.referenceVersion)
    }

    @Test("The VLM uses white RGB and its merged patch order while the VAE retains four straight channels")
    func inputTensorsMatchOriginal() throws {
        let golden = try ImageGolden.load()
        let canonical = try ImageGenerationImageReader.read(NativeGolden.resource("canonical-reference.png"))
        #expect(hash(canonical.rgba) == golden.canonicalSHA256)
        #expect(hash(canonical.whiteRGB()) == golden.whiteSHA256)
        #expect(hash(try canonical.vaeInput()) == golden.vaeSHA256)
        #expect(hash(try canonical.visionInput()) == golden.visionSHA256)
    }

    @Test("RGBA reconstruction clips finite values and preserves ties to even while rejecting invalid input")
    func outputPixelsMatchOriginal() throws {
        let golden = try ImageGolden.load()
        let pattern = golden.decoderPattern
        let values = (0..<(4 * 512 * 512)).map { pattern[$0 % pattern.count] }
        #expect(hash(try ImageGenerationPixels.decodeVAE(values).rgba) == golden.decoderSHA256)
        #expect(throws: ImageGenerationError.invalidShape) { try ImageGenerationPixels.decodeVAE([0]) }
        var nonfinite = values
        nonfinite[19] = .infinity
        #expect(throws: ImageGenerationError.nonFinite) { try ImageGenerationPixels.decodeVAE(nonfinite) }
        let small = try ImageGenerationPixels(width: 1, height: 1, rgba: [255, 0, 255, 0])
        #expect(small.whiteRGB() == [255, 255, 255])
        #expect(try small.resized(width: 1, height: 1).rgba == [255, 0, 255, 0])
        #expect(throws: ImageGenerationError.invalidShape) { try small.vaeInput() }
        #expect(throws: ImageGenerationError.unsupportedImage) { try small.oriented(9) }
    }

    private func hash<Value>(_ values: [Value]) -> String {
        values.withUnsafeBytes { SHA256.hash(data: $0).map { String(format: "%02x", $0) }.joined() }
    }
}

private struct ImageGolden: Decodable {
    struct Transform: Decodable {
        let orientation: Int?
        let width: Int
        let height: Int
        let rgbaSHA256: String
    }

    let sourceSHA256: String
    let orientedSHA256: String
    let orientations: [Transform]
    let resizes: [Transform]
    let canonicalSHA256: String
    let whiteSHA256: String
    let vaeSHA256: String
    let visionSHA256: String
    let decoderPattern: [Float]
    let decoderSHA256: String
    let referenceVersion: String

    static func load() throws -> Self {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: NativeGolden.resource("image-golden.json")))
    }
}
