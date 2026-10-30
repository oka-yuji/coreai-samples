import Foundation
import GLiNERRuntime
import Testing

private func tokenizerFolder(addedID: Int = 3, unknownID: Int = 0) throws -> URL {
    let folder = FileManager.default.temporaryDirectory.appending(path: "gliner-tokenizer-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try Data(
        """
        {"added_tokens": [{"id": \(addedID), "content": "<<ENT>>", "special": true}],
         "model": {"type": "Unigram", "unk_id": \(unknownID), "vocab": [["<unk>", 0.0], ["a", -1.0], ["b", -2.0]]}}
        """.utf8
    ).write(to: folder.appending(path: "tokenizer.json"))
    return folder
}

struct GLiNERTokenizerTests {
    @Test("A tokenizer whose IDs fit in Int32 loads and maps added, known and unknown tokens as before")
    func loadsInRangeIDs() throws {
        let folder = try tokenizerFolder(addedID: 2_147_483_646)
        defer { try? FileManager.default.removeItem(at: folder) }
        let tokenizer = try GLiNERTokenizer(directory: folder)
        #expect(tokenizer.tokenIDs(for: "<<ENT>>") == [2_147_483_646])
        #expect(tokenizer.tokenIDs(for: "a") == [1])
        #expect(tokenizer.tokenIDs(for: "z") == [0])
    }

    @Test(
        "An added token ID outside 0..<Int32.max makes the tokenizer throw instead of ending the process",
        arguments: [2_147_483_647, 4_294_967_296, -1])
    func refusesAddedTokenIDs(id: Int) throws {
        let folder = try tokenizerFolder(addedID: id)
        defer { try? FileManager.default.removeItem(at: folder) }
        #expect(throws: GLiNERTokenizerError.tokenIDOutOfRange(id)) { try GLiNERTokenizer(directory: folder) }
    }

    @Test(
        "An unknown token ID outside Int32 or the vocabulary makes the tokenizer throw instead of ending the process",
        arguments: [4_294_967_296, -1, 3])
    func refusesUnknownTokenIDs(id: Int) throws {
        let folder = try tokenizerFolder(unknownID: id)
        defer { try? FileManager.default.removeItem(at: folder) }
        #expect(throws: GLiNERTokenizerError.tokenIDOutOfRange(id)) { try GLiNERTokenizer(directory: folder) }
    }
}
