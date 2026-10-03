import Foundation
import Tokenizers

public enum BundleError: Error, Equatable, LocalizedError {
    case missingFile(String)
    case missingField(String)
    case invalidValue(String, Int)
    case vocabularyMismatch(tokenizer: Int, metadata: Int)
    case notLanguageModel(outputWidth: Int, vocabSize: Int)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let path): "The model bundle has no \(path)."
        case .missingField(let key): "metadata.json has no \(key)."
        case .invalidValue(let key, let value): "\(key) in metadata.json is \(value); it must be positive."
        case .vocabularyMismatch(let tokenizer, let metadata):
            "The tokenizer has \(tokenizer) tokens, more than language.vocab_size (\(metadata)) in metadata.json."
        case .notLanguageModel(let outputWidth, let vocabSize):
            "The main function returns \(outputWidth) values per token, not \(vocabSize) logits (language.vocab_size); this is not a language model this runtime can run."
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

    public init(url: URL) throws {
        func exists(_ path: String) -> Bool {
            FileManager.default.fileExists(atPath: url.appending(path: path).path(percentEncoded: false))
        }
        guard exists("metadata.json") else { throw BundleError.missingFile("metadata.json") }
        let metadata = try JSONDecoder().decode(Metadata.self, from: Data(contentsOf: url.appending(path: "metadata.json")))
        guard let name = metadata.name else { throw BundleError.missingField("name") }
        guard let main = metadata.assets?.main else { throw BundleError.missingField("assets.main") }
        guard let vocabSize = metadata.language?.vocabSize else { throw BundleError.missingField("language.vocab_size") }
        guard let maxContextLength = metadata.language?.maxContextLength else { throw BundleError.missingField("language.max_context_length") }
        guard metadata.language?.functionMap?["main"]?.isEmpty != true else { throw BundleError.missingField("language.function_map.main") }
        for (key, value) in [("language.vocab_size", vocabSize), ("language.max_context_length", maxContextLength)] where value <= 0 {
            throw BundleError.invalidValue(key, value)
        }
        for path in [main, "tokenizer/tokenizer.json"] where !exists(path) {
            throw BundleError.missingFile(path)
        }
        self.url = url
        self.name = name
        modelURL = url.appending(path: main)
        tokenizerURL = url.appending(path: "tokenizer")
        self.vocabSize = vocabSize
        self.maxContextLength = maxContextLength
    }

    public func loadTokenizer() async throws -> any Tokenizer {
        async let count = Self.tokenCount(tokenizerURL.appending(path: "tokenizer.json"))
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        let tokens = try await count
        guard tokens <= vocabSize else { throw BundleError.vocabularyMismatch(tokenizer: tokens, metadata: vocabSize) }
        return tokenizer
    }

    static func tokenCount(_ url: URL) throws -> Int {
        let file = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
        let vocab = (file["model"] as? [String: Any])?["vocab"]
        let ids = (vocab as? [String: Int]).map { Array($0.values) } ?? Array(0..<((vocab as? [Any])?.count ?? 0))
        let added = (file["added_tokens"] as? [[String: Any]] ?? []).compactMap { $0["id"] as? Int }
        return ((ids + added).max() ?? -1) + 1
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

        let name: String?
        let assets: Assets?
        let language: Language?
    }
}
