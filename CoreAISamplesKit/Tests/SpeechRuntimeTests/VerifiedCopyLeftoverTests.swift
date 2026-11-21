import CryptoKit
import Foundation
import SpeechCore
import Testing

@testable import SpeechAssets
@testable import SpeechRuntime

@Suite(.timeLimit(.minutes(1)))
struct VerifiedCopyLeftoverTests {
    @Test(
        "IRL-01 A leftover temporary copy that holds a locked file is gone after the next open succeeds",
        arguments: [UF_IMMUTABLE, UF_APPEND])
    func removesLockedLeftover(_ flag: Int32) throws {
        let fixture = try LeftoverFixture()
        defer { fixture.remove() }
        try fixture.makeLeftover(locking: flag)
        _ = try fixture.open()
        let entries = try fixture.entries()
        #expect(entries == [LeftoverFixture.key])
    }

    @Test(
        "IRL-02 A leftover temporary copy and the new temporary copy are gone after the next open fails on a changed file"
    )
    func removesLeftoverWhenOpenFails() throws {
        let fixture = try LeftoverFixture()
        defer { fixture.remove() }
        try fixture.makeLeftover(locking: UF_IMMUTABLE)
        try fixture.write("tokenizer-X", to: "tokenizer.json", in: fixture.source)
        #expect(throws: SpeechError.self) { try fixture.open() }
        let entries = try fixture.entries()
        #expect(entries.isEmpty)
    }

    @Test(
        "IRL-03 The verified copy, a copy for another manifest and the lock file stay in place when a leftover is removed"
    )
    func keepsVerifiedCopyAndLock() throws {
        let fixture = try LeftoverFixture()
        defer { fixture.remove() }
        _ = try fixture.open()
        let inodes = try fixture.inodes()
        let other = String(repeating: "0", count: 64)
        try fixture.write("other", to: other + "/irodori.json", in: fixture.copies)
        try fixture.makeLeftover(locking: UF_IMMUTABLE)
        _ = try fixture.open()
        let entries = try fixture.entries()
        #expect(entries == [LeftoverFixture.key, other])
        #expect(try fixture.inodes() == inodes)
        #expect(FileManager.default.fileExists(atPath: fixture.copies.appending(path: ".lock").path))
        #expect(try fixture.read(fixture.copies.appending(path: other + "/irodori.json")) == "other")
    }

    @Test("IRL-01 A file reached through a symbolic link inside a leftover keeps its contents and its flags")
    func keepsLinkedFile() throws {
        let fixture = try LeftoverFixture()
        defer { fixture.remove() }
        let keep = try fixture.makeLockedOutsideFile()
        try fixture.write("partial", to: "irodori.json", in: fixture.leftover)
        try FileManager.default.createSymbolicLink(
            at: fixture.leftover.appending(path: "link"), withDestinationURL: fixture.outside)
        try FileManager.default.createSymbolicLink(
            at: fixture.leftover.appending(path: "file-link"), withDestinationURL: keep)
        _ = try fixture.open()
        let entries = try fixture.entries()
        #expect(entries == [LeftoverFixture.key])
        #expect(try fixture.read(keep) == "keep")
        #expect(try fixture.flags(keep) & UInt32(UF_IMMUTABLE) != 0)
    }

    @Test("IRL-03 A folder whose name only contains the prefix and a symbolic link named like a leftover stay in place")
    func keepsLookalikes() throws {
        let fixture = try LeftoverFixture()
        defer { fixture.remove() }
        let keep = try fixture.makeLockedOutsideFile()
        try fixture.write("note", to: "keep-incoming-1/note.txt", in: fixture.copies)
        try FileManager.default.createSymbolicLink(
            at: fixture.copies.appending(path: "incoming-link"), withDestinationURL: fixture.outside)
        _ = try fixture.open()
        let entries = try fixture.entries()
        #expect(entries == [LeftoverFixture.key, "keep-incoming-1", "incoming-link"])
        #expect(try fixture.read(fixture.copies.appending(path: "keep-incoming-1/note.txt")) == "note")
        #expect(try fixture.read(keep) == "keep")
        #expect(try fixture.flags(keep) & UInt32(UF_IMMUTABLE) != 0)
    }

    @Test("IRL-01 A leftover that cannot be removed does not fail the next open")
    func opensDespiteUnremovableLeftover() throws {
        let fixture = try LeftoverFixture()
        defer { fixture.remove() }
        try fixture.write("partial", to: "irodori.json", in: fixture.leftover)
        let sealed = fixture.leftover.appending(path: "sealed", directoryHint: .isDirectory)
        try fixture.write("inner", to: "inner.txt", in: sealed)
        try #require(chmod(sealed.path, 0o500) == 0)
        defer { chmod(sealed.path, 0o700) }
        #expect(throws: Never.self) { try fixture.open() }
        let entries = try fixture.entries()
        #expect(entries == [LeftoverFixture.key, fixture.leftover.lastPathComponent])
    }
}

private struct LeftoverFixture {
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
