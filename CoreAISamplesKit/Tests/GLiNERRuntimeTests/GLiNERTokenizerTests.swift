import Foundation
import GLiNERRuntime
import Testing

private func tokenizerFolder(addedID: String = "3", unknownID: String = "0") throws -> URL {
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
        let folder = try tokenizerFolder(addedID: "2147483646")
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
        let folder = try tokenizerFolder(addedID: String(id))
        defer { try? FileManager.default.removeItem(at: folder) }
        #expect(throws: GLiNERTokenizerError.tokenIDOutOfRange(String(id))) { try GLiNERTokenizer(directory: folder) }
    }

    @Test(
        "An unknown token ID outside Int32 or the vocabulary makes the tokenizer throw instead of ending the process",
        arguments: [4_294_967_296, -1, 3])
    func refusesUnknownTokenIDs(id: Int) throws {
        let folder = try tokenizerFolder(unknownID: String(id))
        defer { try? FileManager.default.removeItem(at: folder) }
        #expect(throws: GLiNERTokenizerError.tokenIDOutOfRange(String(id))) { try GLiNERTokenizer(directory: folder) }
    }

    @Test(
        "An added or unknown token ID that the parser cannot read as an Int makes the tokenizer throw instead of dropping it",
        arguments: [
            ("9223372036854775808", "9.223372e+18"), ("-9223372036854775809", "-9.223372e+18"),
            ("4294967296.0", "4.2949673e+09"), (#""5""#, #""5""#),
        ])
    func refusesUnreadableTokenIDs(json: String, value: String) throws {
        let folders = try [tokenizerFolder(addedID: json), tokenizerFolder(unknownID: json)]
        defer { for folder in folders { try? FileManager.default.removeItem(at: folder) } }
        for folder in folders {
            #expect(throws: GLiNERTokenizerError.tokenIDOutOfRange(value)) { try GLiNERTokenizer(directory: folder) }
        }
    }
}
