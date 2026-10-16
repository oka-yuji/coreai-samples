import Foundation
import Testing

@testable import EmbeddingCore

@Suite("Embedding preprocessing contracts")
struct EmbeddingCoreTests {
    @Test("Reject invalid dimensions, buffers, and vectors")
    func invalidInputs() {
        #expect(throws: EmbeddingError.invalidImage) { try RGBImage(width: 0, height: 1, pixels: []) }
        #expect(throws: EmbeddingError.invalidImage) { try RGBImage(width: 1, height: 1, pixels: [0]) }
        #expect(throws: EmbeddingError.invalidImage) { try EmbeddingImage.size(width: Int.max, height: 1) }
        #expect(throws: EmbeddingError.invalidVector) { try EmbeddingVector([Float](repeating: 0, count: 768)) }
        #expect(throws: EmbeddingError.invalidVector) { try EmbeddingVector([Float](repeating: .nan, count: 768)) }
    }

    @Test("Match geometry for square, landscape, portrait, and extreme inputs")
    func geometry() throws {
        let square = try EmbeddingImage.size(width: 144, height: 144)
        #expect(square.width == 384 && square.height == 384)
        let wide = try EmbeddingImage.size(width: 320, height: 96)
        #expect(wide.width == 720 && wide.height == 192)
        let tall = try EmbeddingImage.size(width: 96, height: 320)
        #expect(tall.width == wide.height && tall.height == wide.width)
        let narrow = try EmbeddingImage.size(width: 1, height: 10_000)
        #expect(narrow.width == 48 && narrow.height == 3_360)
    }

    @Test("Rotate and mirror pixel positions using EXIF orientation")
    func orientation() throws {
        let image = try RGBImage(
            width: 2, height: 3, pixels: (1...6).flatMap { [UInt8](repeating: UInt8($0), count: 3) })
        let expected: [[UInt8]] = [
            [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5], [6, 5, 4, 3, 2, 1], [5, 6, 3, 4, 1, 2],
            [1, 3, 5, 2, 4, 6], [5, 3, 1, 6, 4, 2], [6, 4, 2, 5, 3, 1], [2, 4, 6, 1, 3, 5],
        ]
        for orientation in 1...8 {
            let rotated = try image.oriented(orientation)
            #expect(
                stride(from: 0, to: rotated.pixels.count, by: 3).map { rotated.pixels[$0] } == expected[orientation - 1]
            )
        }
    }

    @Test("Preserve RGB patch order, spatial pooling, and masked padding")
    func patches() throws {
        var pixels: [UInt8] = []
        for y in 0..<384 {
            for x in 0..<384 { pixels.append(contentsOf: [UInt8(x % 256), UInt8(y % 256), 123]) }
        }
        let image = try RGBImage(width: 384, height: 384, pixels: pixels)
        let prepared = try EmbeddingImage(image)
        #expect(prepared.pixels[0] == 0)
        #expect(abs(prepared.pixels[2] - Float(123) / 255) < 0.000_000_1)
        #expect(prepared.pixels[768] == Float(16) / 255)
        #expect(prepared.positions[2] == 1)
        #expect(prepared.positions[48] == 0 && prepared.positions[49] == 1)
        #expect(prepared.pooling[0] == Float(1) / 9)
        #expect(prepared.pooling[24] == Float(1) / 9)
        #expect(prepared.pooling[3] == 0)
        #expect(prepared.pooling[630 + 3] == Float(1) / 9)
        #expect(prepared.positions.suffix(108).allSatisfy { $0 == -1 })
        #expect(prepared.pixels.suffix(54 * 768).allSatisfy { $0 == 0 })
        #expect(prepared.softTokenCount == 64)
    }
}
