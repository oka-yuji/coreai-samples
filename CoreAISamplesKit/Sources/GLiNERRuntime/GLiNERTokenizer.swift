import Foundation
import Hub
import Tokenizers

/// The mDeBERTa-v3 SentencePiece Unigram tokenizer of GLiNER2.5-multi, read from `tokenizer.json`.
public struct GLiNERTokenizer: Sendable {
    private let tokenizer: PreTrainedTokenizer
    private let addedTokenIDs: [String: Int32]
    private let unknownTokenID: Int32

    public init(directory: URL) throws {
        let data = try HubApi.shared.configuration(fileURL: directory.appending(path: "tokenizer.json"))
        // tokenizer_config.json names DebertaV2Tokenizer, which swift-transformers lacks and would
        // replace with BPE. The model in tokenizer.json is Unigram, which XLMRobertaTokenizer maps to.
        tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: ["tokenizer_class": "XLMRobertaTokenizer"], tokenizerData: data)
        // The Unigram model only knows model.vocab, so [P], [E], [L], [SEP_TEXT] ... would become [UNK].
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
        // The Python tokenizer yields nothing for blank text; swift-transformers' Metaspace yields U+2581.
        guard !word.allSatisfy(\.isWhitespace) else { return [] }
        return tokenizer.tokenize(text: word).map { token in
            addedTokenIDs[token] ?? tokenizer.convertTokenToId(token).map(Int32.init) ?? unknownTokenID
        }
    }
}
