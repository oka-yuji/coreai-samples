import CoreGraphics
import Foundation
import ImageGenerationCore
import ImageGenerationFeature
import ImageGenerationRuntime
import ImageIO
import Testing
import UniformTypeIdentifiers

struct ImageGenerationMediaTests {
    @Test("PNG export preserves every straight RGBA byte and exact seed and model metadata")
    func pngIdentity() async throws {
        let request = ImageGenerationRequest(id: UUID(), prompt: "private prompt", seed: UInt64(Int64.max))
        let result = try imageInteractionResult(request)
        let media = ImageGenerationMedia()
        let asset = try await media.asset(result, request: request)
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "generated.png")
        try asset.png.write(to: url)
        #expect(try ImageGenerationImageReader.read(url) == result.pixels)
        let source = try #require(CGImageSourceCreateWithData(asset.png as CFData, nil))
        let properties = try #require(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
        let png = try #require(properties[kCGImagePropertyPNGDictionary] as? [CFString: Any])
        let description = try #require(png[kCGImagePropertyPNGDescription] as? String)
        let metadata = try #require(JSONSerialization.jsonObject(with: Data(description.utf8)) as? [String: Any])
        #expect(metadata["seed"] as? String == String(Int64.max))
        #expect(metadata["sourceRevision"] as? String == ImageGenerationProfile.revision)
        #expect(metadata["manifestSHA256"] as? String == result.report.manifestSHA256)
        #expect(metadata["steps"] as? Int == 8)
        #expect(!description.contains("private prompt"))
        let before = try Data(contentsOf: url)
        let reference = try await media.reference(url, ticket: ImageGenerationTicket(owner: UUID(), sequence: 1))
        #expect(reference.pixels == result.pixels)
        #expect(try Data(contentsOf: url) == before)
    }

    @Test(
        "Small RGB tagged and oriented sources retain the canonical input bytes before crop confirmation",
        arguments: ["rgba-source", "rgba-oriented", "rgba-srgb"])
    func canonicalReference(_ name: String) async throws {
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "png", subdirectory: "Resources"))
        let before = try Data(contentsOf: url)
        let expected = try ImageGenerationImageReader.read(url).squareReference()
        let preview = try await ImageGenerationMedia().reference(
            url, ticket: ImageGenerationTicket(owner: UUID(), sequence: 1))
        #expect(preview.pixels == expected)
        #expect(try Data(contentsOf: url) == before)
    }

    @Test("Large oriented sources are bounded before the confirmed square is made and the original is preserved")
    func largeReference() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "large.png")
        let rgba = Data((0..<(2_048 * 1_024)).flatMap { _ in [UInt8(64), 128, 192, 255] })
        let provider = try #require(CGDataProvider(data: rgba as CFData))
        let colorSpace = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let image = try #require(
            CGImage(
                width: 2_048, height: 1_024, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: 2_048 * 4, space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try #require(
            CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: 6] as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))
        let original = try Data(contentsOf: url)
        let preview = try await ImageGenerationMedia().reference(
            url, ticket: ImageGenerationTicket(owner: UUID(), sequence: 1))
        let thumbnailSource = try #require(CGImageSourceCreateWithData(preview.thumbnail as CFData, nil))
        let thumbnail = try #require(CGImageSourceCreateImageAtIndex(thumbnailSource, 0, nil))
        #expect(thumbnail.width == 512 && thumbnail.height == 1_024)
        let croppedSource = try #require(CGImageSourceCreateWithData(preview.cropped as CFData, nil))
        let cropped = try #require(CGImageSourceCreateImageAtIndex(croppedSource, 0, nil))
        #expect(try ImageGenerationImageReader.read(cropped) == preview.pixels)
        #expect(preview.pixels.width == 512 && preview.pixels.height == 512)
        #expect(preview.pixels.rgba == Array(rgba.prefix(512 * 512 * 4)))
        #expect(try Data(contentsOf: url) == original)
    }

    @Test("Unselected network locations and incomplete generation results are refused")
    func invalidMedia() async throws {
        let media = ImageGenerationMedia()
        let url = try #require(URL(string: "https://example.invalid/image.png"))
        await #expect(throws: ImageGenerationError.unsupportedImage) {
            try await media.reference(url, ticket: ImageGenerationTicket(owner: UUID(), sequence: 1))
        }
        let request = ImageGenerationRequest(id: UUID(), prompt: "image", seed: 1)
        let other = ImageGenerationRequest(id: UUID(), prompt: "image", seed: 1)
        let result = try imageInteractionResult(other)
        await #expect(throws: ImageExecutionError.invalidResource) { try await media.asset(result, request: request) }
    }
}
