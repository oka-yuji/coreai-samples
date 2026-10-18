import EmbeddingCore
import Foundation

struct TokenEmbeddingTable {
    private let data: Data
    private let scales: Data?
    private let rows: Int

    init(url: URL, scalesURL: URL? = nil) throws {
        try self.init(
            data: Data(contentsOf: url, options: .mappedIfSafe),
            scales: try scalesURL.map { try Data(contentsOf: $0, options: .mappedIfSafe) })
    }

    init(data: Data, scales: Data? = nil, rows: Int = 262_144) throws {
        guard (1...262_144).contains(rows), data.count == rows * (scales == nil ? 1024 : 256) else {
            throw EmbeddingError.invalidBundle("Incorrect embedding table size")
        }
        if let scales {
            guard scales.count == rows * 4 else {
                throw EmbeddingError.invalidBundle("Incorrect INT4 scale table size")
            }
            let valid = scales.withUnsafeBytes { bytes in
                (0..<rows).allSatisfy { row in
                    let bits = UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: row * 4, as: UInt32.self))
                    let scale = Float(bitPattern: bits)
                    return scale.isFinite && scale > 0 && scale <= Float.greatestFiniteMagnitude / 160
                }
            }
            guard valid, !data.contains(where: { $0 & 15 == 8 || $0 >> 4 == 8 }) else {
                throw EmbeddingError.invalidBundle("Invalid symmetric INT4 values or scales")
            }
        }
        self.data = data
        self.scales = scales
        self.rows = rows
    }

    func embeddings(for tokens: [Int]) throws -> [Float] {
        guard tokens.count <= 128 else { throw EmbeddingError.tooManyTokens(tokens.count) }
        for token in tokens where !(0..<rows).contains(token) { throw EmbeddingError.invalidToken(token) }
        let padded = tokens + Array(repeating: 0, count: 128 - tokens.count)
        let normalization = sqrt(Float(512))
        return data.withUnsafeBytes { bytes in
            padded.flatMap { token in
                if let scales {
                    let scale = scales.withUnsafeBytes { scaleBytes in
                        Float(
                            bitPattern: UInt32(
                                littleEndian: scaleBytes.loadUnaligned(fromByteOffset: token * 4, as: UInt32.self)))
                    }
                    return (0..<512).map { column in
                        let packed = Int(bytes[token * 256 + column / 2])
                        let nibble = (packed >> (column % 2 * 4)) & 15
                        return Float(nibble >= 8 ? nibble - 16 : nibble) * scale * normalization
                    }
                }
                return (0..<512).map { column in
                    let bits = UInt16(
                        littleEndian: bytes.loadUnaligned(fromByteOffset: (token * 512 + column) * 2, as: UInt16.self))
                    return Float(bitPattern: UInt32(bits) << 16) * normalization
                }
            }
        }
    }
}
