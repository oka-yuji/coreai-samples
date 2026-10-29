import CImagePNG
import Foundation
import ImageGenerationCore
import ImageIO

enum ImageGenerationPNG {
    private static let encodedLimit = 256 * 1_024 * 1_024

    static func read(_ url: URL) throws -> ImageGenerationPixels {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var data = Data()
        while true {
            try Task.checkCancellation()
            let chunk = try handle.read(upToCount: min(1_048_576, encodedLimit - data.count + 1)) ?? Data()
            guard !chunk.isEmpty else { break }
            guard data.count <= encodedLimit - chunk.count else { throw ImageGenerationError.unsupportedImage }
            data.append(chunk)
        }
        return try decode(data)
    }

    static func decode(_ data: Data) throws -> ImageGenerationPixels {
        guard data.count <= encodedLimit else { throw ImageGenerationError.unsupportedImage }
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithData(data as CFData, options),
            CGImageSourceGetCount(source) == 1,
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let sourceWidth = properties[kCGImagePropertyPixelWidth] as? Int,
            let sourceHeight = properties[kCGImagePropertyPixelHeight] as? Int,
            sourceWidth > 0, sourceHeight > 0, sourceWidth <= 32_768, sourceHeight <= 32_768,
            sourceWidth * sourceHeight <= 40_000_000,
            properties[kCGImagePropertyColorModel] as? String == kCGImagePropertyColorModelRGB as String,
            properties[kCGImagePropertyDepth] as? Int == 8
        else { throw ImageGenerationError.unsupportedImage }
        let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
        if data.count >= 16, data.dropFirst(12).prefix(4).elementsEqual([67, 103, 66, 73]) {
            guard let image = CGImageSourceCreateImageAtIndex(source, 0, options) else {
                throw ImageGenerationError.unsupportedImage
            }
            return try ImageGenerationImageReader.read(image).oriented(orientation)
        }
        var storage: UnsafeMutablePointer<UInt8>?
        var width: UInt32 = 0
        var height: UInt32 = 0
        var count = 0
        let status = data.withUnsafeBytes { bytes in
            coreai_png_decode_rgb8(
                bytes.bindMemory(to: UInt8.self).baseAddress, bytes.count, 32_768, 40_000_000,
                &storage, &width, &height, &count)
        }
        guard status == 0, let storage else { throw ImageGenerationError.unsupportedImage }
        defer { coreai_png_release(storage) }
        guard Int(width) == sourceWidth, Int(height) == sourceHeight,
            count == Int(width) * Int(height) * 4
        else { throw ImageGenerationError.invalidShape }
        try Task.checkCancellation()
        let pixels = try ImageGenerationPixels(
            width: Int(width), height: Int(height),
            rgba: Array(UnsafeBufferPointer(start: storage, count: count)))
        return try pixels.oriented(orientation)
    }
}
