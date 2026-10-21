import CoreGraphics
import CryptoKit
import Foundation
import ImageGenerationCore
import ImageGenerationFeature
import ImageIO
import UniformTypeIdentifiers

public actor ImageGenerationMedia {
    public static let referenceVersion = "imageio-rgb8-max1024-confirmed-center512-v1"

    public init() {}

    public func reference(_ url: URL, ticket: ImageGenerationTicket) throws -> ImageReferencePreview {
        try Task.checkCancellation()
        guard url.isFileURL else { throw ImageGenerationError.unsupportedImage }
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithURL(url as CFURL, options), CGImageSourceGetCount(source) == 1,
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let width = properties[kCGImagePropertyPixelWidth] as? Int,
            let height = properties[kCGImagePropertyPixelHeight] as? Int,
            width > 0, height > 0, width <= 32_768, height <= 32_768, width * height <= 268_435_456
        else { throw ImageGenerationError.unsupportedImage }
        let pixels: ImageGenerationPixels
        if max(width, height) <= 1_024 {
            guard let image = CGImageSourceCreateImageAtIndex(source, 0, options) else {
                throw ImageGenerationError.unsupportedImage
            }
            pixels = try ImageGenerationImageReader.read(image)
                .oriented(properties[kCGImagePropertyOrientation] as? Int ?? 1)
        } else {
            guard
                let image = CGImageSourceCreateThumbnailAtIndex(
                    source, 0,
                    [
                        kCGImageSourceCreateThumbnailFromImageAlways: true,
                        kCGImageSourceCreateThumbnailWithTransform: true,
                        kCGImageSourceThumbnailMaxPixelSize: 1_024,
                        kCGImageSourceShouldCacheImmediately: true,
                    ] as CFDictionary), image.width <= 1_024, image.height <= 1_024
            else {
                throw ImageGenerationError.unsupportedImage
            }
            pixels = try ImageGenerationImageReader.read(image)
        }
        try Task.checkCancellation()
        let cropped = try pixels.squareReference()
        let preview = try ImageReferencePreview(
            id: ticket, thumbnail: png(pixels), cropped: png(cropped), pixels: cropped)
        try Task.checkCancellation()
        return preview
    }

    public func asset(_ result: ImageGenerationResult, request: ImageGenerationRequest) throws -> ImageGeneratedAsset {
        try Task.checkCancellation()
        guard result.requestID == request.id, result.report.requestID == request.id, result.report.status == .completed,
            result.pixels.width == 512, result.pixels.height == 512
        else {
            throw ImageExecutionError.invalidResource
        }
        var metadata: [String: Any] = [
            "model": "Qwen/Qwen-Image-2.1-Turbo", "sourceRevision": ImageGenerationProfile.revision,
            "profileID": result.report.profileID, "manifestSHA256": result.report.manifestSHA256,
            "seed": String(request.seed), "width": result.pixels.width, "height": result.pixels.height,
            "steps": 8, "cfg": 1, "noiseSource": result.report.noiseSource,
        ]
        if let reference = request.reference {
            metadata["referenceVersion"] = Self.referenceVersion
            metadata["referenceSHA256"] = SHA256.hash(data: Data(reference.rgba)).map { String(format: "%02x", $0) }
                .joined()
        }
        let description = String(
            decoding: try JSONSerialization.data(withJSONObject: metadata, options: .sortedKeys), as: UTF8.self)
        let data = try png(result.pixels, description: description)
        try Task.checkCancellation()
        return ImageGeneratedAsset(id: request.id, png: data, seed: request.seed, report: result.report)
    }

    public func join() {}

    private func png(_ pixels: ImageGenerationPixels, description: String? = nil) throws -> Data {
        let bytes = Data(pixels.rgba)
        guard let provider = CGDataProvider(data: bytes as CFData),
            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
            let image = CGImage(
                width: pixels.width, height: pixels.height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: pixels.width * 4, space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { throw ImageGenerationError.unsupportedImage }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw ImageGenerationError.unsupportedImage
        }
        let properties: CFDictionary? = description.map {
            [kCGImagePropertyPNGDictionary: [kCGImagePropertyPNGDescription: $0]] as CFDictionary
        }
        CGImageDestinationAddImage(destination, image, properties)
        guard CGImageDestinationFinalize(destination) else { throw ImageGenerationError.unsupportedImage }
        return data as Data
    }
}
