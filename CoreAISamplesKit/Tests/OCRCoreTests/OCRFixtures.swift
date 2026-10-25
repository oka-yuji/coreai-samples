import CryptoKit
import Foundation
import OCRCore
import Testing

enum OCRFixtures {
    static func load<Value: Decodable>(_ name: String, as type: Value.Type = Value.self) throws -> Value {
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Resources"))
        return try JSONDecoder().decode(Value.self, from: Data(contentsOf: url))
    }

    static func pattern(width: Int, height: Int) throws -> OCRImage {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                for channel in 0..<3 {
                    let seed = UInt64(x * 1_103 + y * 2_927 + channel * 4_111 + x * y * 17) * 2_654_435_761
                    rgb[(y * width + x) * 3 + channel] = UInt8((seed % (1 << 32)) >> 24)
                }
            }
        }
        return try OCRImage(width: width, height: height, rgb: rgb)
    }

    static func digest(_ bytes: [UInt8]) -> String {
        SHA256.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined()
    }

    static func digest(_ values: [Float]) -> String {
        values.withUnsafeBytes { SHA256.hash(data: Data($0)) }.map { String(format: "%02x", $0) }.joined()
    }

    static func floats(_ bits: [UInt32]) -> [Float] { bits.map(Float.init(bitPattern:)) }

    static func maximumDifference(_ left: some Collection<Float>, _ right: some Collection<Float>) -> Float {
        zip(left, right).map { abs($0 - $1) }.max() ?? 0
    }
}

struct RotaryRow: Decodable {
    let index: Int
    let cosineBits: [UInt32]
    let sineBits: [UInt32]
}

struct VisionFixture: Decodable {
    struct PixelRow: Decodable {
        let index: Int
        let values: [UInt32]
    }

    struct Rotary: Decodable {
        let inverseFrequencyBits: [UInt32]
        let firstAxisPositions: [Int]
        let secondAxisPositions: [Int]
        let cosineTableBits: [UInt32]
        let sineTableBits: [UInt32]
        let rows: [RotaryRow]
    }

    struct Case: Decodable {
        let name: String
        let width: Int
        let height: Int
        let grid: [Int]
        let pixelValuesSHA256: String
        let windowLengths: [Int]
        let pixelRows: [PixelRow]?
        let windowIndex: [Int]?
        let restore: [Int]?
        let rotary: Rotary?
    }

    let cases: [Case]
}

struct PromptFixture: Decodable {
    struct Segment: Decodable {
        let text: String
        let ids: [Int32]
    }

    struct Case: Decodable {
        let name: String
        let prompt: String
        let gridHeight: Int
        let gridWidth: Int
        let ids: [Int32]
        let imageStart: Int
        let imageCount: Int
        let positions: [[Int]]
        let ropeDelta: Int
    }

    struct DecodeRow: Decodable {
        let writePosition: Int
        let cosineBits: [UInt32]
        let sineBits: [UInt32]
    }

    struct TextRotary: Decodable {
        let inverseFrequencyBits: [UInt32]
        let sectionAxes: [Int]
        let frequencyIndex: [Int]
        let cosineTableBits: [UInt32]
        let sineTableBits: [UInt32]
        let rows: [RotaryRow]
        let decode: [DecodeRow]
    }

    let segments: [Segment]
    let cases: [Case]
    let textRotary: TextRotary
}
