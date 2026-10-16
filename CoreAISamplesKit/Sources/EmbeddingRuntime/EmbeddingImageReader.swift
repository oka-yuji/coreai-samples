import CoreGraphics
import EmbeddingCore
import Foundation
import ImageIO

public enum EmbeddingImageReader {
    public static func read(_ url: URL) throws -> RGBImage {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            CGImageSourceGetCount(source) == 1,
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let width = properties[kCGImagePropertyPixelWidth] as? Int,
            let height = properties[kCGImagePropertyPixelHeight] as? Int,
            width > 0, height > 0, width <= 32_768, height <= 32_768, width * height <= 40_000_000,
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw EmbeddingError.invalidImage }
        return try read(image).oriented(properties[kCGImagePropertyOrientation] as? Int ?? 1)
    }

    public static func read(_ image: CGImage) throws -> RGBImage {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0, width <= 32_768, height <= 32_768, width * height <= 40_000_000 else {
            throw EmbeddingError.invalidImage
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let untagged: CGImage
        if image.colorSpace?.model == .rgb {
            guard let copy = image.copy(colorSpace: colorSpace) else { throw EmbeddingError.invalidImage }
            untagged = copy
        } else {
            untagged = image
        }
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = rgba.withUnsafeMutableBytes { bytes -> Bool in
            guard
                let context = CGContext(
                    data: bytes.baseAddress, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: width * 4, space: colorSpace,
                    bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            context.draw(untagged, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw EmbeddingError.invalidImage }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        try rgba.withUnsafeBufferPointer { input in
            try rgb.withUnsafeMutableBufferPointer { output in
                for pixel in 0..<(width * height) {
                    let source = pixel * 4
                    guard input[source + 3] == 255 else { throw EmbeddingError.invalidImage }
                    let destination = pixel * 3
                    output[destination] = input[source]
                    output[destination + 1] = input[source + 1]
                    output[destination + 2] = input[source + 2]
                }
            }
        }
        return try RGBImage(width: width, height: height, pixels: rgb)
    }
}
