import CryptoKit
import Foundation
import SpeechCore
import Testing

@testable import SpeechAssets
@testable import SpeechRuntime

struct VerifiedCopyLeftoverFixture {
    static let original = [
        "irodori.json": "manifest",
        "tokenizer.json": "tokenizer-1",
        "artifacts/a.aimodel/main.mlirb": "model-a-1",
    ]
    static let key = digest("manifest")

    let base = FileManager.default.temporaryDirectory.appending(
        path: "VerifiedCopyLeftoverTests-" + UUID().uuidString, directoryHint: .isDirectory)
    var source: URL { base.appending(path: "source", directoryHint: .isDirectory) }
    var copies: URL { base.appending(path: "copies", directoryHint: .isDirectory) }
    var outside: URL { base.appending(path: "outside", directoryHint: .isDirectory) }
    var leftover: URL {
        copies.appending(path: "incoming-00000000-0000-0000-0000-000000000000", directoryHint: .isDirectory)
    }

    init() throws {
        for (path, text) in Self.original { try write(text, to: path, in: source) }
    }

    static func digest(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    static func layout(_ manifest: Data) throws -> SpeechBundleLayout {
        guard manifest == Data("manifest".utf8) else { throw SpeechError.invalidAsset("Unknown test manifest") }
        return SpeechBundleLayout(
            files: original.map { path, text in
                IrodoriBundleInventory.File(path: path, bytes: Int64(text.utf8.count), sha256: digest(text))
            }, assets: ["artifacts/a.aimodel"])
    }

    func open() throws -> ValidatedSpeechBundle {
        try ValidatedSpeechBundle.open(source, copyingInto: copies, layout: Self.layout)
    }

    func makeLeftover(locking flag: Int32) throws {
        try write("partial", to: "irodori.json", in: leftover)
        try write("model", to: "artifacts/a.aimodel/main.mlirb", in: leftover)
        try #require(chflags(leftover.appending(path: "artifacts/a.aimodel/main.mlirb").path, UInt32(flag)) == 0)
    }

    func makeLockedOutsideFile() throws -> URL {
        let keep = outside.appending(path: "keep.txt")
        try write("keep", to: "keep.txt", in: outside)
        try #require(chflags(keep.path, UInt32(UF_IMMUTABLE)) == 0)
        return keep
    }

    func write(_ text: String, to path: String, in folder: URL) throws {
        let url = folder.appending(path: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }

    func read(_ url: URL) throws -> String {
        try String(contentsOf: url, encoding: .utf8)
    }

    func flags(_ url: URL) throws -> UInt32 {
        var status = stat()
        try #require(lstat(url.path, &status) == 0)
        return status.st_flags
    }

    func inodes() throws -> [String: UInt64] {
        var result: [String: UInt64] = [:]
        for path in Self.original.keys {
            var status = stat()
            try #require(lstat(copies.appending(path: Self.key + "/" + path).path, &status) == 0)
            result[path] = status.st_ino
        }
        return result
    }

    func entries() throws -> Set<String> {
        Set(try FileManager.default.contentsOfDirectory(atPath: copies.path)).subtracting([".lock"])
    }

    func remove() {
        for path in (try? FileManager.default.subpathsOfDirectory(atPath: base.path)) ?? [] {
            lchflags(base.appending(path: path).path, 0)
        }
        try? FileManager.default.removeItem(at: base)
    }
}
