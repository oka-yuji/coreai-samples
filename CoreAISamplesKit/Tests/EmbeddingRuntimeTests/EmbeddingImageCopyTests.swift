import CoreGraphics
import EmbeddingCore
import EmbeddingRuntime
import Foundation
import Testing

struct EmbeddingImageCopyTests {
    @Test(
        "Optimized RGB extraction preserves opaque pixels and rejects transparent images", arguments: [1, 3, 33, 128])
    func pixelsAndAlpha(width: Int) throws {
        let height = width + 2
        var rgba: [UInt8] = []
        var expected: [UInt8] = []
        for pixel in 0..<(width * height) {
            let rgb = [
                UInt8(truncatingIfNeeded: pixel * 17), UInt8(truncatingIfNeeded: pixel * 31),
                UInt8(truncatingIfNeeded: pixel * 7),
            ]
            rgba.append(contentsOf: rgb + [255])
            expected.append(contentsOf: rgb)
        }
        let opaque = try image(width: width, height: height, rgba: rgba)
        let result = try EmbeddingImageReader.read(opaque)
        #expect(result.width == width && result.height == height && result.pixels == expected)
        for alpha: UInt8 in [0, 128, 254] {
            rgba[rgba.count - 1] = alpha
            let transparent = try image(width: width, height: height, rgba: rgba)
            #expect(throws: EmbeddingError.invalidImage) { try EmbeddingImageReader.read(transparent) }
        }
    }

    private func image(width: Int, height: Int, rgba: [UInt8]) throws -> CGImage {
        let provider = try #require(CGDataProvider(data: Data(rgba) as CFData))
        return try #require(
            CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    }
}
