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
