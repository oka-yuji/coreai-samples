import Foundation
import SpeechCore
import Testing

@Suite(.timeLimit(.minutes(1)))
struct VerifiedCopyLeftoverTests {
    @Test(
        "IRL-01 A leftover temporary copy that holds a locked file is gone after the next open succeeds",
        arguments: [UF_IMMUTABLE, UF_APPEND])
    func removesLockedLeftover(_ flag: Int32) throws {
        let fixture = try VerifiedCopyLeftoverFixture()
        defer { fixture.remove() }
        try fixture.makeLeftover(locking: flag)
        _ = try fixture.open()
        let entries = try fixture.entries()
        #expect(entries == [VerifiedCopyLeftoverFixture.key])
    }

    @Test(
        "IRL-02 A leftover temporary copy and the new temporary copy are gone after the next open fails on a changed file"
    )
    func removesLeftoverWhenOpenFails() throws {
        let fixture = try VerifiedCopyLeftoverFixture()
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
        let fixture = try VerifiedCopyLeftoverFixture()
        defer { fixture.remove() }
        _ = try fixture.open()
        let inodes = try fixture.inodes()
        let other = String(repeating: "0", count: 64)
        try fixture.write("other", to: other + "/irodori.json", in: fixture.copies)
        try fixture.makeLeftover(locking: UF_IMMUTABLE)
        _ = try fixture.open()
        let entries = try fixture.entries()
        #expect(entries == [VerifiedCopyLeftoverFixture.key, other])
        #expect(try fixture.inodes() == inodes)
        #expect(FileManager.default.fileExists(atPath: fixture.copies.appending(path: ".lock").path))
        #expect(try fixture.read(fixture.copies.appending(path: other + "/irodori.json")) == "other")
    }

    @Test("IRL-01 A file reached through a symbolic link inside a leftover keeps its contents and its flags")
    func keepsLinkedFile() throws {
        let fixture = try VerifiedCopyLeftoverFixture()
        defer { fixture.remove() }
        let keep = try fixture.makeLockedOutsideFile()
        try fixture.write("partial", to: "irodori.json", in: fixture.leftover)
        try FileManager.default.createSymbolicLink(
            at: fixture.leftover.appending(path: "link"), withDestinationURL: fixture.outside)
        try FileManager.default.createSymbolicLink(
            at: fixture.leftover.appending(path: "file-link"), withDestinationURL: keep)
        _ = try fixture.open()
        let entries = try fixture.entries()
        #expect(entries == [VerifiedCopyLeftoverFixture.key])
        #expect(try fixture.read(keep) == "keep")
        #expect(try fixture.flags(keep) & UInt32(UF_IMMUTABLE) != 0)
    }

    @Test("IRL-03 A folder whose name only contains the prefix and a symbolic link named like a leftover stay in place")
    func keepsLookalikes() throws {
        let fixture = try VerifiedCopyLeftoverFixture()
        defer { fixture.remove() }
        let keep = try fixture.makeLockedOutsideFile()
        try fixture.write("note", to: "keep-incoming-1/note.txt", in: fixture.copies)
        try FileManager.default.createSymbolicLink(
            at: fixture.copies.appending(path: "incoming-link"), withDestinationURL: fixture.outside)
        _ = try fixture.open()
        let entries = try fixture.entries()
        #expect(entries == [VerifiedCopyLeftoverFixture.key, "keep-incoming-1", "incoming-link"])
        #expect(try fixture.read(fixture.copies.appending(path: "keep-incoming-1/note.txt")) == "note")
        #expect(try fixture.read(keep) == "keep")
        #expect(try fixture.flags(keep) & UInt32(UF_IMMUTABLE) != 0)
    }

    @Test("IRL-01 A leftover that cannot be removed does not fail the next open")
    func opensDespiteUnremovableLeftover() throws {
        let fixture = try VerifiedCopyLeftoverFixture()
        defer { fixture.remove() }
        try fixture.write("partial", to: "irodori.json", in: fixture.leftover)
        let sealed = fixture.leftover.appending(path: "sealed", directoryHint: .isDirectory)
        try fixture.write("inner", to: "inner.txt", in: sealed)
        try #require(chmod(sealed.path, 0o500) == 0)
        defer { chmod(sealed.path, 0o700) }
        #expect(throws: Never.self) { try fixture.open() }
        let entries = try fixture.entries()
        #expect(entries == [VerifiedCopyLeftoverFixture.key, fixture.leftover.lastPathComponent])
    }
}
