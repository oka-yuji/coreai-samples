import Foundation
import Hub
import Tokenizers

public enum BundleError: Error, Equatable, LocalizedError {
    case missingFile(String)
    case missingField(String)
    case invalidValue(String, Int)
    case vocabularyMismatch(tokenizer: Int, metadata: Int)
    case notLanguageModel(outputWidth: Int, vocabSize: Int)
    case tokenIDOutOfRange(Int)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let path): "The model bundle has no \(path)."
        case .missingField(let key): "metadata.json has no \(key)."
        case .invalidValue(let key, let value):
            "\(key) in metadata.json is \(value); it must be \(value > 0 ? "at most \(Int32.max)" : "positive")."
        case .vocabularyMismatch(let tokenizer, let metadata):
            "The tokenizer has \(tokenizer) tokens, more than language.vocab_size (\(metadata)) in metadata.json."
        case .notLanguageModel(let outputWidth, let vocabSize):
            "The main function returns \(outputWidth) values per token, not \(vocabSize) logits (language.vocab_size); this is not a language model this runtime can run."
        case .tokenIDOutOfRange(let id):
            "tokenizer.json has the token ID \(id); token IDs must be from 0 to \(Int32.max - 1)."
        }
    }
}

public struct ModelBundle: Sendable {
    public let url: URL
    public let name: String
    public let modelURL: URL
    public let tokenizerURL: URL
    public let vocabSize: Int
    public let maxContextLength: Int
    public let sourceModelID: String?
    public let mainHash: String?

    public init(url: URL) throws {
        func exists(_ path: String) -> Bool {
            FileManager.default.fileExists(atPath: url.appending(path: path).path(percentEncoded: false))
        }
        guard exists("metadata.json") else { throw BundleError.missingFile("metadata.json") }
        let metadata = try JSONDecoder().decode(
            Metadata.self, from: Data(contentsOf: url.appending(path: "metadata.json")))
        guard let name = metadata.name else { throw BundleError.missingField("name") }
        guard let main = metadata.assets?.main else { throw BundleError.missingField("assets.main") }
        guard let vocabSize = metadata.language?.vocabSize else {
            throw BundleError.missingField("language.vocab_size")
        }
        guard let maxContextLength = metadata.language?.maxContextLength else {
            throw BundleError.missingField("language.max_context_length")
        }
        guard metadata.language?.functionMap?["main"]?.isEmpty != true else {
            throw BundleError.missingField("language.function_map.main")
        }
        for (key, value) in [("language.vocab_size", vocabSize), ("language.max_context_length", maxContextLength)]
        where value <= 0 {
            throw BundleError.invalidValue(key, value)
        }
        guard vocabSize <= Int32.max else { throw BundleError.invalidValue("language.vocab_size", vocabSize) }
        for path in [main, "tokenizer/tokenizer.json"] where !exists(path) {
            throw BundleError.missingFile(path)
        }
        self.url = url
        self.name = name
        modelURL = url.appending(path: main)
        tokenizerURL = url.appending(path: "tokenizer")
        self.vocabSize = vocabSize
        self.maxContextLength = maxContextLength
        sourceModelID = metadata.source?.hfModelID
        let hash = try? Data(contentsOf: modelURL.appending(path: "main.hash"))
        mainHash = hash?.map { String(format: "%02x", $0) }.joined()
    }

    public func loadTokenizer() async throws -> any Tokenizer {
        async let count = Self.tokenCount(tokenizerURL.appending(path: "tokenizer.json"))
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        let tokens = try await count
        guard tokens <= vocabSize else { throw BundleError.vocabularyMismatch(tokenizer: tokens, metadata: vocabSize) }
        return tokenizer
    }

    func addedTokenStrings() throws -> [String] {
        let data = try Data(contentsOf: tokenizerURL.appending(path: "tokenizer.json"))
        return try JSONDecoder().decode(TokenizerMetadata.self, from: data).addedTokens?.map(\.content) ?? []
    }

    private struct TokenizerMetadata: Decodable {
        struct Token: Decodable {
            let content: String
        }

        let addedTokens: [Token]?

        enum CodingKeys: String, CodingKey {
            case addedTokens = "added_tokens"
        }
    }

    static func tokenCount(_ url: URL) throws -> Int {
        let file = try HubApi.shared.configuration(fileURL: url)
        let vocab: Config = file.model.vocab
        let ids =
            (vocab.dictionary()?.values.compactMap { $0.integer() } ?? Array(vocab.array(or: []).indices))
            + file.addedTokens.array(or: []).compactMap { $0["id"].integer() }
        if let id = ids.first(where: { !(0..<Int(Int32.max)).contains($0) }) {
            throw BundleError.tokenIDOutOfRange(id)
        }
        return (ids.max() ?? -1) + 1
    }

    private struct Metadata: Decodable {
        struct Assets: Decodable {
            let main: String?
        }

        struct Language: Decodable {
            let vocabSize: Int?
            let maxContextLength: Int?
            let functionMap: [String: [String]]?

            enum CodingKeys: String, CodingKey {
                case vocabSize = "vocab_size"
                case maxContextLength = "max_context_length"
                case functionMap = "function_map"
            }
        }

        struct Source: Decodable {
            let hfModelID: String?

            enum CodingKeys: String, CodingKey {
                case hfModelID = "hf_model_id"
            }

            init(from decoder: any Swift.Decoder) {
                hfModelID = try? decoder.container(keyedBy: CodingKeys.self).decode(String.self, forKey: .hfModelID)
            }
        }

        let name: String?
        let assets: Assets?
        let language: Language?
        let source: Source?
    }
}
