import Foundation
import Hub
import Tokenizers

public enum GLiNERTokenizerError: Error, Equatable, LocalizedError {
    case tokenIDOutOfRange(Int)

    public var errorDescription: String? {
        switch self {
        case .tokenIDOutOfRange(let id):
            "tokenizer.json has the token ID \(id), outside the vocabulary or the range 0 to \(Int32.max - 1)."
        }
    }
}

public struct GLiNERTokenizer: Sendable {
    private let tokenizer: PreTrainedTokenizer
    private let addedTokenIDs: [String: Int32]
    private let unknownTokenID: Int32

    public init(directory: URL) throws {
        let data = try HubApi.shared.configuration(fileURL: directory.appending(path: "tokenizer.json"))
        let vocabulary = data.model.vocab.array(or: []).indices
        let ids = data.addedTokens.array(or: []).compactMap { $0["id"].integer() } + vocabulary.suffix(1)
        if let id = ids.first(where: { !(0..<Int(Int32.max)).contains($0) }) {
            throw GLiNERTokenizerError.tokenIDOutOfRange(id)
        }
        if let id = data.model.unkId.integer(), !vocabulary.contains(id) {
            throw GLiNERTokenizerError.tokenIDOutOfRange(id)
        }

        tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: ["tokenizer_class": "XLMRobertaTokenizer"], tokenizerData: data)

        var addedTokenIDs: [String: Int32] = [:]
        for token in data.addedTokens.array(or: []) {
            if let content = token.content.string(), let id = token["id"].integer() {
                addedTokenIDs[content] = Int32(id)
            }
        }
        self.addedTokenIDs = addedTokenIDs
        unknownTokenID = Int32(tokenizer.unknownTokenId ?? 0)
    }

    public func tokenIDs(for word: String) -> [Int32] {

        guard !word.allSatisfy(\.isWhitespace) else { return [] }
        return tokenizer.tokenize(text: word).map { token in
            addedTokenIDs[token] ?? tokenizer.convertTokenToId(token).map(Int32.init) ?? unknownTokenID
        }
    }
}
