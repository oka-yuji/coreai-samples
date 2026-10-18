import EmbeddingCore
import Foundation
import Testing

@testable import EmbeddingRuntime

@Suite("Packed vocabulary compatibility")
struct TokenEmbeddingTableTests {
    @Test("INT4 rows preserve signed nibble order, row scale, and padding")
    func quantizedRows() throws {
        let data = Data(repeating: 0x97, count: 256) + Data(repeating: 0x1F, count: 256)
        let table = try TokenEmbeddingTable(data: data, scales: floats([0.5, 2]), rows: 2)
        let values = try table.embeddings(for: [1, 0])
        let factor = sqrt(Float(512))
        #expect(values.count == 128 * 512)
        #expect(values[0] == -2 * factor)
        #expect(values[1] == 2 * factor)
        #expect(values[512] == 3.5 * factor)
        #expect(values[513] == -3.5 * factor)
        #expect(values[1024] == values[512])
        #expect(throws: EmbeddingError.invalidToken(2)) { try table.embeddings(for: [2]) }
        #expect(throws: EmbeddingError.invalidToken(-1)) { try table.embeddings(for: [-1]) }
        #expect(throws: EmbeddingError.tooManyTokens(129)) {
            try table.embeddings(for: Array(repeating: 0, count: 129))
        }
    }

    @Test("BF16 vocabulary rows retain their original scaling")
    func legacyRows() throws {
        let data = Data(Array(repeating: [UInt8(0x80), 0x3F], count: 512).flatMap { $0 })
        let table = try TokenEmbeddingTable(data: data, rows: 1)
        #expect(try table.embeddings(for: [0]) == Array(repeating: sqrt(Float(512)), count: 128 * 512))
    }

    @Test("Invalid quantized scales and reserved INT4 values are rejected")
    func invalidTables() {
        let data = Data(repeating: 0, count: 256)
        for scale: Float in [0, -1, .nan, .infinity, .greatestFiniteMagnitude] {
            #expect(throws: (any Error).self) { try TokenEmbeddingTable(data: data, scales: floats([scale]), rows: 1) }
        }
        #expect(throws: (any Error).self) { try TokenEmbeddingTable(data: data, scales: Data(), rows: 1) }
        #expect(throws: (any Error).self) {
            try TokenEmbeddingTable(data: data.dropLast(), scales: floats([1]), rows: 1)
        }
        for byte: UInt8 in [0x08, 0x80] {
            #expect(throws: (any Error).self) {
                try TokenEmbeddingTable(data: Data(repeating: byte, count: 256), scales: floats([1]), rows: 1)
            }
        }
    }

    private func floats(_ values: [Float]) -> Data {
        Data(
            values.flatMap { value in
                let bits = value.bitPattern
                return (0..<4).map { UInt8(truncatingIfNeeded: bits >> ($0 * 8)) }
            })
    }
}
