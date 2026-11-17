import Foundation
import SpeechCore
import Testing

@testable import SpeechAssets
@testable import SpeechRuntime

struct VoxCPMVerifiedKeyTests {
    @Test("VML-07 Two generations with one manifest and equal sizes get separate copies and the second reads its bytes")
    func secondGeneration() throws {
        let fixture = try VoxCPMBundleFixture()
        defer { fixture.remove() }
        let second = fixture.contents.mapValues { $0 == fixture.manifest ? $0 : Data($0.reversed()) }
        try #require(second.mapValues(\.count) == fixture.contents.mapValues(\.count))
        try #require(second.filter { fixture.contents[$0.key] != $0.value }.count == fixture.contents.count - 1)
        let folder = fixture.base.appending(path: "second", directoryHint: .isDirectory)
        try write(second, into: folder)
        let first = try open(fixture.source, fixture.contents, fixture)
        let copied = try read(first.directory)
        try #require(copied == fixture.contents)
        let reopened = try open(folder, second, fixture)
        #expect(reopened.directory != first.directory)
        #expect(try read(reopened.directory) == second)
        #expect(try read(first.directory) == copied)
        #expect(try fixture.copyFolders() == [first.directory.lastPathComponent, reopened.directory.lastPathComponent])
    }

    @Test("VML-07 Reopening one generation reuses its copy and the key does not depend on the inventory order")
    func sameGeneration() throws {
        let fixture = try VoxCPMBundleFixture()
        defer { fixture.remove() }
        let first = try open(fixture.source, fixture.contents, fixture)
        let again = try open(fixture.source, fixture.contents, fixture)
        #expect(again.directory == first.directory && first.directory == fixture.copy)
        #expect(try fixture.copyFolders() == [fixture.key])
        let files = try VoxCPMBundleInventory.files(inventory: fixture.inventory())
        for order in [Array(files.reversed()), Array(files.dropFirst(5) + files.prefix(5))] {
            #expect(
                ValidatedSpeechBundle.verifiedKey(manifest: fixture.manifest, files: order, names: .voxcpm2)
                    == fixture.key)
        }
    }

    @Test(
        "VML-07 Swapping paths and hashes or changing only the tokenizer or only the NOTICE changes the key",
        arguments: ["swap", "tokenizer", "notice"])
    func keyChanges(_ change: String) throws {
        let fixture = try VoxCPMBundleFixture()
        defer { fixture.remove() }
        let files = try VoxCPMBundleInventory.files(inventory: fixture.inventory())
        let paths = files.map(\.path)
        var changed = files
        switch change {
        case "swap":
            let first = try #require(paths.firstIndex(of: "artifacts/dit-fp32.aimodel/main.mlirb"))
            let second = try #require(paths.firstIndex(of: "artifacts/vae-fp32.aimodel/main.mlirb"))
            try #require(files[first].bytes == files[second].bytes && files[first].sha256 != files[second].sha256)
            changed[first] = entry(files[first], sha256: files[second].sha256)
            changed[second] = entry(files[second], sha256: files[first].sha256)
        default:
            let index = try #require(paths.firstIndex(of: change == "tokenizer" ? "tokenizer.json" : "NOTICE"))
            changed[index] = entry(files[index], sha256: VoxCPMBundleFixture.digest(Data("changed".utf8)))
        }
        try #require(changed != files)
        #expect(
            ValidatedSpeechBundle.verifiedKey(manifest: fixture.manifest, files: changed, names: .voxcpm2)
                != fixture.key)
    }

    @Test("VML-07 Irodori copies stay keyed by the manifest SHA-256 and VoxCPM2 copies by a length-prefixed file list")
    func keyFormula() {
        let manifest = Data("manifest".utf8)
        let digest = VoxCPMBundleFixture.digest(manifest)
        let notice = VoxCPMBundleFixture.digest(Data("notice".utf8))
        let files = [
            IrodoriBundleInventory.File(path: "manifest.json", bytes: 8, sha256: digest),
            IrodoriBundleInventory.File(path: "NOTICE", bytes: 6, sha256: notice),
        ]
        #expect(ValidatedSpeechBundle.verifiedKey(manifest: manifest, files: files, names: .irodori) == digest)
        var expected = Data()
        for field in ["voxcpm2-verified-v1", "NOTICE", "6", notice, "manifest.json", "8", digest] {
            withUnsafeBytes(of: UInt64(field.utf8.count).bigEndian) { expected.append(contentsOf: $0) }
            expected.append(contentsOf: field.utf8)
        }
        #expect(
            ValidatedSpeechBundle.verifiedKey(manifest: manifest, files: files, names: .voxcpm2)
                == VoxCPMBundleFixture.digest(expected))
    }

    private func entry(_ file: IrodoriBundleInventory.File, sha256: String) -> IrodoriBundleInventory.File {
        IrodoriBundleInventory.File(path: file.path, bytes: file.bytes, sha256: sha256)
    }

    private func open(
        _ source: URL, _ contents: [String: Data], _ fixture: VoxCPMBundleFixture
    ) throws -> ValidatedSpeechBundle {
        let files = try VoxCPMBundleInventory.files(inventory: VoxCPMBundleFixture.inventory(contents))
        return try ValidatedSpeechBundle.open(
            source, copyingInto: fixture.copies,
            layout: { try SpeechBundleLayout.voxcpm2(manifest: $0, files: files) }, names: .voxcpm2,
            inPlaceWhenCloneFails: false)
    }

    private func write(_ contents: [String: Data], into folder: URL) throws {
        for (path, data) in contents {
            let url = folder.appending(path: path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url)
        }
    }

    private func read(_ directory: URL) throws -> [String: Data] {
        var result: [String: Data] = [:]
        for path in try FileManager.default.subpathsOfDirectory(atPath: directory.path) {
            var status = stat()
            let url = directory.appending(path: path)
            if lstat(url.path, &status) == 0, status.st_mode & S_IFMT == S_IFREG {
                result[path] = try Data(contentsOf: url)
            }
        }
        return result
    }
}
