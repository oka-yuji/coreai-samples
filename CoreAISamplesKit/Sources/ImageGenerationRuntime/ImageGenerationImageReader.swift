import CoreGraphics
import Foundation
import ImageGenerationCore
import ImageIO

public enum ImageGenerationImageReader {
    public static func read(_ url: URL) throws -> ImageGenerationPixels {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), CGImageSourceGetCount(source) == 1,
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let width = properties[kCGImagePropertyPixelWidth] as? Int,
            let height = properties[kCGImagePropertyPixelHeight] as? Int,
            width > 0, height > 0, width <= 32_768, height <= 32_768, width * height <= 40_000_000,
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw ImageGenerationError.unsupportedImage }
        return try read(image).oriented(properties[kCGImagePropertyOrientation] as? Int ?? 1)
    }

    public static func read(_ image: CGImage) throws -> ImageGenerationPixels {
        let width = image.width
        let height = image.height
        let channels = image.bitsPerPixel / 8
        guard width > 0, height > 0, width <= 32_768, height <= 32_768, width * height <= 40_000_000,
            image.bitsPerComponent == 8, image.bitsPerPixel.isMultiple(of: 8),
            image.colorSpace?.model == .rgb, (3...4).contains(channels),
            !image.bitmapInfo.contains(.floatComponents),
            let provider = image.dataProvider, let data = provider.data,
            let bytes = CFDataGetBytePtr(data), CFDataGetLength(data) >= image.bytesPerRow * height,
            image.bytesPerRow >= width * channels
        else { throw ImageGenerationError.unsupportedImage }
        let little = image.bitmapInfo.intersection(.byteOrderMask) == .byteOrder32Little
        let alpha = image.alphaInfo
        guard
            alpha == .last || alpha == .first || alpha == .none || alpha == .noneSkipFirst
                || alpha == .noneSkipLast || alpha == .premultipliedFirst || alpha == .premultipliedLast
        else { throw ImageGenerationError.unsupportedImage }
        let leading = alpha == .first || alpha == .noneSkipFirst || alpha == .premultipliedFirst
        let hasAlpha = alpha == .first || alpha == .last || alpha == .premultipliedFirst || alpha == .premultipliedLast
        let premultiplied = alpha == .premultipliedFirst || alpha == .premultipliedLast
        guard channels == 4 || (!hasAlpha && !leading && !little) else { throw ImageGenerationError.unsupportedImage }
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let source = y * image.bytesPerRow + x * channels
                func byte(_ channel: Int) -> UInt8 { bytes[source + (little ? 3 - channel : channel)] }
                let opacity = hasAlpha ? byte(leading ? 0 : 3) : 255
                let destination = (y * width + x) * 4
                rgba[destination + 3] = opacity
                for channel in 0..<3 {
                    let value = byte(channel + (leading ? 1 : 0))
                    rgba[destination + channel] =
                        premultiplied && opacity > 0 && opacity < 255
                        ? UInt8(clamping: (Int(value) * 255 + Int(opacity) / 2) / Int(opacity)) : value
                }
            }
        }
        return try ImageGenerationPixels(width: width, height: height, rgba: rgba)
    }
}
