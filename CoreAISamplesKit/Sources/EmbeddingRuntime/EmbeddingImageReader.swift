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
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let untagged = image.copy(colorSpace: colorSpace) else { throw EmbeddingError.invalidImage }
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
        var rgb = [UInt8]()
        rgb.reserveCapacity(width * height * 3)
        for offset in stride(from: 0, to: rgba.count, by: 4) {
            guard rgba[offset + 3] == 255 else { throw EmbeddingError.invalidImage }
            rgb.append(contentsOf: rgba[offset..<(offset + 3)])
        }
        return try RGBImage(width: width, height: height, pixels: rgb)
            .oriented(properties[kCGImagePropertyOrientation] as? Int ?? 1)
    }
}
