import CImagePNG
import CryptoKit
import Foundation
import Testing

struct ImagePNGDecoderBoundaryTests {
    @Test("The native PNG bridge preserves unassociated samples and rejects smaller decode budgets")
    func dimensionsAndPixels() throws {
        let input = try Data(contentsOf: NativeGolden.resource("transparent-edit-source.png"))
        let accepted = decode(input, dimension: 512, pixels: 512 * 512)
        #expect(accepted.status == 0)
        #expect(accepted.width == 512 && accepted.height == 512)
        let rgba = try #require(accepted.rgba)
        #expect(
            SHA256.hash(data: rgba).map { String(format: "%02x", $0) }.joined()
                == "1ba143d368861e3014b5aa142c751e64e24707f6d8be743fe76049799bf948e2")
        for limits in [(UInt32(511), 512 * 512), (512, 512 * 512 - 1), (0, 512 * 512), (512, 0)] {
            let result = decode(input, dimension: limits.0, pixels: limits.1)
            #expect(result.status != 0)
            #expect(result.width == 0 && result.height == 0 && result.rgba == nil)
        }
    }

    @Test("PNG corruption truncation and bytes after IEND cannot become model input")
    func malformedInput() throws {
        let input = try Data(contentsOf: NativeGolden.resource("transparent-edit-source.png"))
        var corrupted = input
        corrupted[32] ^= 1
        for invalid in [Data(input.prefix(20)), Data(input.dropLast()), corrupted, input + Data([0])] {
            let result = decode(invalid, dimension: 512, pixels: 512 * 512)
            #expect(result.status != 0)
            #expect(result.rgba == nil)
        }
    }

    private func decode(_ data: Data, dimension: UInt32, pixels: Int) -> (
        status: UInt32, width: UInt32, height: UInt32, rgba: Data?
    ) {
        var output: UnsafeMutablePointer<UInt8>?
        var width: UInt32 = 0
        var height: UInt32 = 0
        var count = 0
        let status = data.withUnsafeBytes { bytes in
            coreai_png_decode_rgb8(
                bytes.bindMemory(to: UInt8.self).baseAddress, bytes.count, dimension, pixels,
                &output, &width, &height, &count)
        }
        defer { coreai_png_release(output) }
        return (status, width, height, output.map { Data(bytes: $0, count: count) })
    }
}
