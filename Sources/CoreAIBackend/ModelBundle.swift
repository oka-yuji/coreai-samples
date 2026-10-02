import Foundation
import Tokenizers

public struct ModelBundle: Sendable {
    public let url: URL
    public let name: String
    public let modelURL: URL
    public let tokenizerURL: URL
    public let vocabSize: Int
    public let maxContextLength: Int

    public init(url: URL) throws {
        let metadata = try JSONDecoder().decode(Metadata.self, from: Data(contentsOf: url.appending(path: "metadata.json")))
        self.url = url
        name = metadata.name
        modelURL = url.appending(path: metadata.assets.main)
        tokenizerURL = url.appending(path: "tokenizer")
        vocabSize = metadata.language.vocabSize
        maxContextLength = metadata.language.maxContextLength
    }

    public func loadTokenizer() async throws -> any Tokenizer {
        try await AutoTokenizer.from(modelFolder: tokenizerURL)
    }

    private struct Metadata: Decodable {
        struct Assets: Decodable {
            let main: String
        }

        struct Language: Decodable {
            let vocabSize: Int
            let maxContextLength: Int

            enum CodingKeys: String, CodingKey {
                case vocabSize = "vocab_size"
                case maxContextLength = "max_context_length"
            }
        }

        let name: String
        let assets: Assets
        let language: Language
    }
}
