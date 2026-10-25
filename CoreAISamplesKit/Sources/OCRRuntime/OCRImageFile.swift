import CoreGraphics
import Foundation
import ImageIO
import OCRCore

extension OCRImage {
    public init(data: Data) throws {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw OCRError.invalidImage }
        let width = image.width
        let height = image.height
        let gray = image.colorSpace?.model == .monochrome
        let colors = gray ? 1 : 3
        let alpha = image.alphaInfo
        let channels = colors + (alpha == .none ? 0 : 1)
        let leading = alpha == .first || alpha == .noneSkipFirst || alpha == .premultipliedFirst
        let premultiplied = alpha == .premultipliedFirst || alpha == .premultipliedLast
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        let reversed = order == .byteOrder32Little || order == .byteOrder16Little
        guard image.bitsPerComponent == 8, image.bitsPerPixel == channels * 8, alpha != .alphaOnly,
            gray || image.colorSpace?.model == .rgb, !image.bitmapInfo.contains(.floatComponents),
            image.bytesPerRow >= width * channels, let data = image.dataProvider?.data,
            CFDataGetLength(data) >= image.bytesPerRow * height, let bytes = CFDataGetBytePtr(data)
        else { throw OCRError.invalidImage }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                let base = y * image.bytesPerRow + x * channels
                func byte(_ index: Int) -> Int { Int(bytes[base + (reversed ? channels - 1 - index : index)]) }
                let opacity = premultiplied ? byte(leading ? 0 : colors) : 255
                for channel in 0..<3 {
                    let value = byte((leading ? 1 : 0) + (gray ? 0 : channel))
                    rgb[(y * width + x) * 3 + channel] =
                        opacity == 0 || opacity == 255
                        ? UInt8(value) : UInt8(min(255, (value * 255 + opacity / 2) / opacity))
                }
            }
        }
        try self.init(width: width, height: height, rgb: rgb)
    }
}
