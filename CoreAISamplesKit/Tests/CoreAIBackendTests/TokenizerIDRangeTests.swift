import Foundation
import Testing

@testable import CoreAIBackend

private let addedToken = #"{"id": 3, "content": "<|im_end|>", "special": true}"#

private func tokenizerJSON(vocab: String = #"{"a": 0, "b": 1, "ab": 2}"#, added: String = addedToken) -> String {
    #"{"added_tokens": [\#(added)], "model": {"type": "BPE", "vocab": \#(vocab), "merges": ["a b"]}}"#
}

private let outOfRange: [(String, Int)] = [
    (tokenizerJSON(vocab: #"{"a": 0, "b": 4294967296, "ab": 2}"#), 4_294_967_296),
    (tokenizerJSON(vocab: #"{"a": 0, "b": 9223372036854775807, "ab": 2}"#), Int.max),
    (tokenizerJSON(added: #"{"id": 2147483647, "content": "<|im_end|>", "special": true}"#), 2_147_483_647),
    (tokenizerJSON(added: #"{"id": -1, "content": "<|im_end|>", "special": true}"#), -1),
    (tokenizerJSON(vocab: #"{"a": 0, "b": 1, "ab": 2, "b": 4294967296}"#), 4_294_967_296),
    (
        #"{"addedTokens": [{"id": 4294967296, "content": "<|im_end|>"}], "added_tokens": [\#(addedToken)], "#
            + #""model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}}"#,
        4_294_967_296
    ),
]

private func decisionTokenizer(_ json: String) throws -> URL {
    let folder = FileManager.default.temporaryDirectory.appending(path: "decision-tokenizer-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data(#"{"tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|im_end|>"}"#.utf8)
        .write(to: folder.appending(path: "tokenizer_config.json"))
    try Data(json.utf8).write(to: folder.appending(path: "tokenizer.json"))
    return folder
}

struct TokenizerIDRangeTests {
    @Test(
        "A language model bundle whose tokenizer.json has an ID outside 0..<Int32.max throws when the tokenizer loads",
        arguments: outOfRange)
    func languageModelRefusesTokenIDs(json: String, id: Int) async throws {
        let folder = try tinyBundle()
        defer { try? FileManager.default.removeItem(at: folder) }
        try Data(json.utf8).write(to: folder.appending(path: "tokenizer/tokenizer.json"))
        await #expect(throws: BundleError.tokenIDOutOfRange(id)) { try await ModelBundle(url: folder).loadTokenizer() }
    }

    @Test("A language model bundle whose language.vocab_size is outside Int32 throws when it opens")
    func languageModelRefusesVocabularySize() throws {
        let tooLarge = try tinyBundle(metadata: tinyMetadata.replacingOccurrences(of: "151936", with: "2147483648"))
        let largest = try tinyBundle(metadata: tinyMetadata.replacingOccurrences(of: "151936", with: "2147483647"))
        defer { for folder in [tooLarge, largest] { try? FileManager.default.removeItem(at: folder) } }
        #expect(throws: BundleError.invalidValue("language.vocab_size", 2_147_483_648)) {
            try ModelBundle(url: tooLarge)
        }
        #expect(try ModelBundle(url: largest).vocabSize == 2_147_483_647)
        #expect(
            BundleError.invalidValue("language.vocab_size", 2_147_483_648).localizedDescription
                == "language.vocab_size in metadata.json is 2147483648; it must be at most 2147483647.")
        #expect(
            BundleError.invalidValue("language.vocab_size", 0).localizedDescription
                == "language.vocab_size in metadata.json is 0; it must be positive.")
    }

    @Test(
        "The decision model's tokenizer load throws for an ID outside 0..<Int32.max", arguments: outOfRange)
    func decisionRefusesTokenIDs(json: String, id: Int) async throws {
        let folder = try decisionTokenizer(json)
        defer { try? FileManager.default.removeItem(at: folder) }
        await #expect(throws: BundleError.tokenIDOutOfRange(id)) { try await DecisionModel.loadTokenizer(folder) }
    }

    @Test("The decision model's tokenizer load reads a tokenizer whose IDs fit in Int32 as before")
    func decisionLoadsInRangeTokenizer() async throws {
        let folder = try decisionTokenizer(tokenizerJSON())
        defer { try? FileManager.default.removeItem(at: folder) }
        let tokenizer = try await DecisionModel.loadTokenizer(folder)
        #expect(tokenizer.encode(text: "ab<|im_end|>", addSpecialTokens: false) == [2, 3])
        #expect(tokenizer.eosTokenId == 3)
    }
}
