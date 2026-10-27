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

    @Test(
        "CMYK and sixteen-bit sources are refused before either decode path while the same RGB source is accepted",
        arguments: [512, 2_048])
    func sourceColorModels(_ width: Int) async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let media = ImageGenerationMedia()
        let ticket = ImageGenerationTicket(owner: UUID(), sequence: 1)
        let rgb = try source("rgb.jpg", in: root, width: width, space: CGColorSpace.sRGB, bits: 8, type: .jpeg)
        #expect(rgb.model == kCGImagePropertyColorModelRGB as String && rgb.depth == 8)
        let preview = try await media.reference(rgb.url, ticket: ticket)
        #expect(preview.pixels.width == 512 && preview.pixels.height == 512)
        let cmyk = try source("cmyk.jpg", in: root, width: width, space: CGColorSpace.genericCMYK, bits: 8, type: .jpeg)
        #expect(cmyk.model == kCGImagePropertyColorModelCMYK as String && cmyk.depth == 8)
        await #expect(throws: ImageGenerationError.unsupportedImage) {
            try await media.reference(cmyk.url, ticket: ticket)
        }
        let deep = try source("rgb16.png", in: root, width: width, space: CGColorSpace.sRGB, bits: 16, type: .png)
        #expect(deep.model == kCGImagePropertyColorModelRGB as String && deep.depth == 16)
        await #expect(throws: ImageGenerationError.unsupportedImage) {
            try await media.reference(deep.url, ticket: ticket)
        }
    }

    @Test(
        "Indexed PNG sources are refused at both sizes with or without palette transparency or a leading CgBI chunk",
        arguments: [512, 2_048],
        [
            (transparency: false, cgbi: false), (transparency: true, cgbi: false), (transparency: false, cgbi: true),
            (transparency: true, cgbi: true),
        ])
    func indexedSources(_ width: Int, _ variant: (transparency: Bool, cgbi: Bool)) async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "indexed.png")
        try indexedPNG(width: width, transparency: variant.transparency, cgbi: variant.cgbi).write(to: url)
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let properties = try #require(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
        #expect(properties[kCGImagePropertyColorModel] as? String == kCGImagePropertyColorModelRGB as String)
        #expect(properties[kCGImagePropertyDepth] as? Int == 8)
        #expect(CGImageSourceCreateImageAtIndex(source, 0, nil)?.width == width)
        await #expect(throws: ImageGenerationError.unsupportedImage) {
            try await ImageGenerationMedia().reference(url, ticket: ImageGenerationTicket(owner: UUID(), sequence: 1))
        }
    }

    @Test("Truncated PNG files are refused", arguments: [8, 20])
    func truncatedPNG(_ count: Int) async throws {
        let fixture = try #require(
            Bundle.module.url(forResource: "rgba-source", withExtension: "png", subdirectory: "Resources"))
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "truncated.png")
        try Data(contentsOf: fixture).prefix(count).write(to: url)
        await #expect(throws: ImageGenerationError.unsupportedImage) {
            try await ImageGenerationMedia().reference(url, ticket: ImageGenerationTicket(owner: UUID(), sequence: 1))
        }
    }

    @Test("A CgBI RGBA source is decoded in RGB channel order and accepted")
    func appleRGBA() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "cgbi.png")
        try appleRGBAPNG(width: 512).write(to: url)
        let preview = try await ImageGenerationMedia().reference(
            url, ticket: ImageGenerationTicket(owner: UUID(), sequence: 1))
        #expect(preview.pixels.width == 512 && preview.pixels.height == 512)
        #expect(preview.pixels.rgba == Array([[UInt8]](repeating: [64, 128, 192, 255], count: 512 * 512).joined()))
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

    private func source(
        _ name: String, in root: URL, width: Int, space: CFString, bits: Int, type: UTType
    ) throws -> (url: URL, model: String?, depth: Int?) {
        let url = root.appending(path: name)
        let colorSpace = try #require(CGColorSpace(name: space))
        let alpha = space == CGColorSpace.genericCMYK ? CGImageAlphaInfo.none : .noneSkipLast
        let context = try #require(
            CGContext(
                data: nil, width: width, height: 512, bitsPerComponent: bits, bytesPerRow: 0, space: colorSpace,
                bitmapInfo: alpha.rawValue))
        context.setFillColor(CGColor(red: 0.2, green: 0.5, blue: 0.8, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: 512))
        let image = try #require(context.makeImage())
        let destination = try #require(
            CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        let imageSource = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let properties = try #require(CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [CFString: Any])
        return (url, properties[kCGImagePropertyColorModel] as? String, properties[kCGImagePropertyDepth] as? Int)
    }

    private func indexedPNG(width: Int, transparency: Bool, cgbi: Bool) throws -> Data {
        let rows: [UInt8] = (0..<512).flatMap { _ in [0] + (0..<width).map { UInt8($0 % 2) } }
        var low: UInt32 = 1
        var high: UInt32 = 0
        for byte in rows {
            low = (low + UInt32(byte)) % 65_521
            high = (high + low) % 65_521
        }
        let deflated = try (Data(rows) as NSData).compressed(using: .zlib) as Data
        var png: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        if cgbi { png += chunk("CgBI", [0x50, 0x00, 0x20, 0x02]) }
        png += chunk("IHDR", bigEndian(UInt32(width)) + bigEndian(512) + [8, 3, 0, 0, 0])
        png += chunk("PLTE", [64, 128, 192, 192, 128, 64])
        if transparency { png += chunk("tRNS", [255, 128]) }
        png += chunk("IDAT", [0x78, 0x01] + Array(deflated) + bigEndian(high << 16 | low))
        png += chunk("IEND", [])
        return Data(png)
    }

    private func appleRGBAPNG(width: Int) throws -> Data {
        let rows: [UInt8] = (0..<512).flatMap { _ in [0] + (0..<width).flatMap { _ in [192, 128, 64, 255] } }
        let deflated = try (Data(rows) as NSData).compressed(using: .zlib) as Data
        var png: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        png += chunk("CgBI", [0x50, 0x00, 0x20, 0x02])
        png += chunk("IHDR", bigEndian(UInt32(width)) + bigEndian(512) + [8, 6, 0, 0, 0])
        png += chunk("IDAT", Array(deflated))
        png += chunk("IEND", [])
        return Data(png)
    }

    private func bigEndian(_ value: UInt32) -> [UInt8] { withUnsafeBytes(of: value.bigEndian, Array.init) }

    private func chunk(_ name: String, _ body: [UInt8]) -> [UInt8] {
        let named = Array(name.utf8) + body
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in named {
            crc ^= UInt32(byte)
            for _ in 0..<8 { crc = crc & 1 == 1 ? (crc >> 1) ^ 0xEDB8_8320 : crc >> 1 }
        }
        return bigEndian(UInt32(body.count)) + named + bigEndian(~crc)
    }
}
