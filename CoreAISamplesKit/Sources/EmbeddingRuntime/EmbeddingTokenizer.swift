import EmbeddingCore
import Foundation
import Hub
import Tokenizers

public struct EmbeddingTokenizer: Sendable {
    private let tokenizer: PreTrainedTokenizer

    public init(directory: URL) throws {
        tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: HubApi.shared.configuration(fileURL: directory.appending(path: "tokenizer_config.json")),
            tokenizerData: HubApi.shared.configuration(fileURL: directory.appending(path: "tokenizer.json")))
    }

    public func tokens(for text: String, role: EmbeddingTextRole) throws -> [Int] {
        guard text.utf8.count <= 32_768 else { throw EmbeddingError.tooManyTokens(129) }
        let tokens = tokenizer.encode(text: role.prefix + text, addSpecialTokens: true)
        guard tokens.count <= 128 else { throw EmbeddingError.tooManyTokens(tokens.count) }
        guard tokens.first == 2, tokens.last == 1 else {
            throw EmbeddingError.invalidBundle("Tokenizer must insert BOS and EOS")
        }
        return tokens
    }
}

struct TokenEmbeddingTable {
    private let data: Data

    init(url: URL) throws {
        data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count == 262_144 * 512 * 2 else {
            throw EmbeddingError.invalidBundle("Incorrect BF16 embedding table size")
        }
    }

    func embeddings(for tokens: [Int]) throws -> [Float] {
        guard tokens.count <= 128 else { throw EmbeddingError.tooManyTokens(tokens.count) }
        for token in tokens where !(0..<262_144).contains(token) { throw EmbeddingError.invalidToken(token) }
        let padded = tokens + Array(repeating: 0, count: 128 - tokens.count)
        return data.withUnsafeBytes { bytes in
            padded.flatMap { token in
                (0..<512).map { column in
                    let bits = UInt16(
                        littleEndian: bytes.loadUnaligned(fromByteOffset: (token * 512 + column) * 2, as: UInt16.self))
                    return Float(bitPattern: UInt32(bits) << 16) * sqrt(Float(512))
                }
            }
        }
    }
}
