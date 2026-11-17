import Foundation
import Testing
import VoxCPMCore
import VoxCPMTokenizers

struct VoxCPMTokenizerTests {
    struct Reference: Decodable {
        struct Prompt: Decodable {
            let name: String
            let text: String
            let tokenIDs: [Int32]
            let expectedIDs: [Int32]
        }

        let prompts: [Prompt]
    }

    @Test(
        "VIN-01 For the fixture sentences the wrapper and the kanji split give the original IDs from the full vocabulary",
        .enabled(if: ProcessInfo.processInfo.environment["VOXCPM2_TOKENIZER"] != nil))
    func fixtureSentences() throws {
        let folder = try #require(ProcessInfo.processInfo.environment["VOXCPM2_TOKENIZER"])
        let tokenizer = try VoxCPMTokenizer(tokenizerJSON: URL(filePath: folder).appending(path: "tokenizer.json"))
        let fixture = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "VoxCPMCoreTests/Resources/reference.json")
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: fixture))
        #expect(reference.prompts.count == 4)
        for prompt in reference.prompts {
            let ids = try tokenizer.tokenIDs(for: prompt.text)
            #expect(ids == prompt.tokenIDs, "\(prompt.name)")
            let split = try VoxCPMPrompt.inputIDs(textTokenIDs: ids, vocabulary: tokenizer.vocabulary)
            #expect(split == prompt.expectedIDs, "\(prompt.name)")
        }
        #expect(tokenizer.vocabulary.count == 73_448)
        let semicolon = try #require(tokenizer.vocabulary[59_375])
        let questionMark = try #require(tokenizer.vocabulary[70_138])
        #expect(semicolon == questionMark)
        #expect(Array(semicolon.unicodeScalars) != Array(questionMark.unicodeScalars))
        let groups = Dictionary(grouping: tokenizer.vocabulary.keys) {
            tokenizer.vocabulary[$0]?.precomposedStringWithCanonicalMapping ?? ""
        }
        #expect(groups.values.filter { $0.count > 1 }.count == 13)
    }

    @Test("VIN-02 A tokenizer.json whose SHA-256 differs from the pinned value is rejected")
    func rejectsOtherTokenizer() throws {
        let folder = FileManager.default.temporaryDirectory.appending(
            path: "VoxCPMTokenizerTests-" + UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let file = folder.appending(path: "tokenizer.json")
        try Data(#"{"model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}}"#.utf8).write(to: file)
        #expect(throws: VoxCPMTokenizerError.unexpectedSHA256) { try VoxCPMTokenizer(tokenizerJSON: file) }
        #expect(throws: (any Error).self) {
            try VoxCPMTokenizer(tokenizerJSON: folder.appending(path: "missing.json"))
        }
    }
}
