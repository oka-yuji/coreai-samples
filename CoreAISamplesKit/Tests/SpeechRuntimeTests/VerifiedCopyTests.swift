import CryptoKit
import Foundation
import SpeechCore
import Synchronization
import Testing

@testable import SpeechAssets
@testable import SpeechRuntime

@Suite(.timeLimit(.minutes(1)))
struct VerifiedCopyTests {
    @Test("SPK-01 Generation and reloads after release, a failure or a length change read only the verified bytes")
    func readsVerifiedCopy() async throws {
        let fixture = try BundleFixture()
        defer { fixture.remove() }
        let driver = TestDriver()
        let reads = LifetimeLog()
        let runtime = IrodoriRuntime(dependencies: fixture.dependencies(driver, reads: reads))
        let model = try await runtime.open(fixture.source)
        #expect(model.directory == fixture.source)
        for (path, text) in BundleFixture.replacements { try fixture.overwrite(path, with: text) }
        _ = try await runtime.generate(text: "13", seed: 1)
        await runtime.release()
        _ = try await runtime.generate(text: "13", seed: 2)
        await driver.configure(fail: true)
        await #expect(throws: SpeechError.invalidSamples) { try await runtime.generate(text: "13", seed: 3) }
        await driver.configure()
        _ = try await runtime.generate(text: "13", seed: 4)
        _ = try await runtime.generate(text: "26", seed: 5)
        #expect(await driver.factoryCount == 4)
        #expect(try fixture.copyFolders() == [BundleFixture.key])
        let tokenizer = "\(fixture.copy.appending(path: "tokenizer.json").path)=tokenizer-1"
        let assets = ["a", "b"].map { "\(fixture.copy.appending(path: "artifacts/\($0).aimodel").path)=model-\($0)-1" }
        #expect(reads.snapshot() == [tokenizer] + assets + [tokenizer] + assets + assets + assets)
        #expect(
            try String(contentsOf: fixture.source.appending(path: "tokenizer.json"), encoding: .utf8) == "tokenizer-2")
    }

    @Test("SPK-01 The open checks the bytes of the copy rather than the source")
    func checksCopiedBytes() async throws {
        let fixture = try BundleFixture()
        defer { fixture.remove() }
        let altered: @Sendable (Int32, String) -> Int32 = { descriptor, path in
            let code = ValidatedSpeechBundle.cloneFile(descriptor, to: path)
            if code == 0, path.hasSuffix("/tokenizer.json") {
                try? Data("tokenizer-X".utf8).write(to: URL(filePath: path))
            }
            return code
        }
        let reads = LifetimeLog()
        let runtime = IrodoriRuntime(dependencies: fixture.dependencies(TestDriver(), reads: reads, clone: altered))
        await #expect(throws: SpeechError.self) { try await runtime.open(fixture.source) }
        #expect(try fixture.copyFolders().isEmpty)
        #expect(reads.snapshot().isEmpty)
    }

    @Test(
        "SPK-02 A symlinked, FIFO, changed or missing source file fails the open and leaves no copy",
        arguments: SourceDefect.allCases)
    func rejectsUnsafeSource(_ defect: SourceDefect) async throws {
        let fixture = try BundleFixture()
        defer { fixture.remove() }
        try fixture.apply(defect)
        let driver = TestDriver()
        let reads = LifetimeLog()
        let runtime = IrodoriRuntime(dependencies: fixture.dependencies(driver, reads: reads))
        await #expect(throws: SpeechError.self) { try await runtime.open(fixture.source) }
        #expect(try fixture.copyFolders().isEmpty)
        #expect(reads.snapshot().isEmpty)
        await #expect(throws: SpeechError.noModel) { try await runtime.generate(text: "13", seed: 1) }
        #expect(await driver.factoryCount == 0)
    }

    @Test(
        "SPK-02 The pinned manifest selects the fixed inventory and a folder without model files leaves no copy",
        arguments: [SpeechBundleProfile.textOnly, .referenceVoice])
    func pinnedManifest(_ profile: SpeechBundleProfile) throws {
        let fields = [
            "\"format\": \"\(profile.rawValue)\"", "\"modelID\": \"Aratako/Irodori-TTS-v4.1-Small-MF\"",
            "\"precision\": \"fp32\"", "\"sampleRate\": 48000",
        ]
        let manifest: Data
        if profile == .referenceVoice {
            let url = try #require(
                Bundle.module.url(
                    forResource: "release-reference-manifest", withExtension: "json", subdirectory: "Resources"))
            manifest = try Data(contentsOf: url)
        } else {
            manifest = Data(("{\n  " + fields.joined(separator: ",\n  ") + "\n}\n").utf8)
        }
        let layout = try SpeechBundleLayout.irodori(manifest: manifest)
        let stems =
            profile == .textOnly
            ? ["text-256-fp32-upstream", "duration-no-ref-fp32", "dit-dynamic13-750-fp32", "codec-dynamic13-750-fp32"]
            : [
                "text-256-fp32-upstream", "duration-reference-fp32", "dit-reference-fp32", "codec-dynamic13-750-fp32",
                "reference-audio-fp32", "reference-speaker-fp32",
            ]
        #expect(layout.assets == stems.map { "artifacts/\($0).aimodel" })
        #expect(layout.files == (try IrodoriBundleInventory.files(profile: profile)))
        #expect(throws: (any Error).self) { try SpeechBundleLayout.irodori(manifest: manifest + Data(" ".utf8)) }
        let fixture = try BundleFixture()
        defer { fixture.remove() }
        try manifest.write(to: fixture.source.appending(path: "irodori.json"))
        let missing = try #require(layout.files.first { $0.path != "irodori.json" }).path
        let message = "Model files do not match the verified Irodori FP32 version: " + missing
        #expect(throws: SpeechError.invalidAsset(message)) {
            try ValidatedSpeechBundle.open(fixture.source, copyingInto: fixture.copies)
        }
        #expect(try fixture.copyFolders().isEmpty)
    }

    @Test("SPK-02 Files outside the inventory are not copied")
    func copiesOnlyInventory() async throws {
        let fixture = try BundleFixture()
        defer { fixture.remove() }
        for path in ["extra.txt", "artifacts/a.aimodel/weights.bin", "artifacts/c.aimodel/main.mlirb"] {
            try fixture.write("extra", to: path)
        }
        let runtime = IrodoriRuntime(dependencies: fixture.dependencies(TestDriver(), reads: LifetimeLog()))
        _ = try await runtime.open(fixture.source)
        #expect(try Set(fixture.inodes().keys) == Set(BundleFixture.original.keys))
    }

    @Test("SPK-03 Later opens, other runtimes and new runtimes reuse the same copied files for the same content")
    func reusesVerifiedCopy() async throws {
        let fixture = try BundleFixture()
        defer { fixture.remove() }
        let firstReads = LifetimeLog()
        var first: IrodoriRuntime? = IrodoriRuntime(
            dependencies: fixture.dependencies(TestDriver(), reads: firstReads))
        _ = try await first?.open(fixture.source)
        _ = try await first?.generate(text: "13", seed: 1)
        let inodes = try fixture.inodes()
        #expect(inodes.count == BundleFixture.original.count)
        _ = try await first?.open(fixture.source)
        #expect(try fixture.copyFolders() == [BundleFixture.key])
        #expect(try fixture.inodes() == inodes)
        let secondReads = LifetimeLog()
        let second = IrodoriRuntime(dependencies: fixture.dependencies(TestDriver(), reads: secondReads))
        _ = try await second.open(fixture.source)
        _ = try await second.generate(text: "13", seed: 1)
        #expect(secondReads.snapshot() == Array(firstReads.snapshot().prefix(3)))
        first = nil
        let duplicate = try fixture.duplicateSource()
        let restarted = IrodoriRuntime(dependencies: fixture.dependencies(TestDriver(), reads: LifetimeLog()))
        _ = try await restarted.open(duplicate)
        #expect(try fixture.copyFolders() == [BundleFixture.key])
        #expect(try fixture.inodes() == inodes)
    }

    @Test(
        "SPK-03 A copy with a missing or resized file is replaced by a newly verified copy", arguments: [false, true])
    func replacesDamagedCopy(_ resized: Bool) async throws {
        let fixture = try BundleFixture()
        defer { fixture.remove() }
        let runtime = IrodoriRuntime(dependencies: fixture.dependencies(TestDriver(), reads: LifetimeLog()))
        _ = try await runtime.open(fixture.source)
        let inodes = try fixture.inodes()
        let damaged = fixture.copy.appending(path: "artifacts/a.aimodel/main.mlirb")
        if resized {
            let handle = try FileHandle(forWritingTo: damaged)
            try handle.seekToEnd()
            try handle.write(contentsOf: Data("!".utf8))
            try handle.close()
        } else {
            try FileManager.default.removeItem(at: damaged)
        }
        let reads = LifetimeLog()
        let repairing = IrodoriRuntime(dependencies: fixture.dependencies(TestDriver(), reads: reads))
        _ = try await repairing.open(fixture.source)
        _ = try await repairing.generate(text: "13", seed: 1)
        #expect(try fixture.copyFolders() == [BundleFixture.key])
        let replaced = try fixture.inodes()
        #expect(Set(replaced.keys) == Set(inodes.keys))
        #expect(replaced["tokenizer.json"] != inodes["tokenizer.json"])
        let assets = ["a", "b"].map { "\(fixture.copy.appending(path: "artifacts/\($0).aimodel").path)=model-\($0)-1" }
        #expect(Array(reads.snapshot().suffix(2)) == assets)
    }

    @Test("SPK-03 A locked source file leaves no temporary copy when the verified copy is reused")
    func lockedSource() async throws {
        let fixture = try BundleFixture()
        defer { fixture.remove() }
        let locked = fixture.source.appending(path: "tokenizer.json").path
        try #require(chflags(locked, UInt32(UF_IMMUTABLE)) == 0)
        defer { chflags(locked, 0) }
        let runtime = IrodoriRuntime(dependencies: fixture.dependencies(TestDriver(), reads: LifetimeLog()))
        _ = try await runtime.open(fixture.source)
        _ = try await runtime.open(fixture.source)
        #expect(try fixture.copyFolders() == [BundleFixture.key])
    }

    @Test(
        "SPK-03 A source file locked between its check and its clone leaves no temporary copy",
        arguments: [false, true])
    func lockedBeforeClone(_ changed: Bool) async throws {
        let fixture = try BundleFixture()
        defer { fixture.remove() }
        _ = try ValidatedSpeechBundle.open(fixture.source, copyingInto: fixture.copies, layout: BundleFixture.layout)
        if changed { try fixture.apply(.changedBytes) }
        let locked = fixture.source.appending(path: "tokenizer.json").path
        defer { chflags(locked, 0) }
        let lockThenClone: @Sendable (Int32, String) -> Int32 = { descriptor, path in
            if path.hasSuffix("/tokenizer.json") { fchflags(descriptor, UInt32(UF_IMMUTABLE)) }
            return ValidatedSpeechBundle.cloneFile(descriptor, to: path)
        }
        let runtime = IrodoriRuntime(
            dependencies: fixture.dependencies(TestDriver(), reads: LifetimeLog(), clone: lockThenClone))
        if changed {
            await #expect(throws: SpeechError.self) { try await runtime.open(fixture.source) }
        } else {
            _ = try await runtime.open(fixture.source)
        }
        #expect(try FileManager.default.attributesOfItem(atPath: locked)[.immutable] as? Bool == true)
        #expect(try fixture.copyFolders() == [BundleFixture.key])
    }

    @Test("SPK-03 Two opens that repair the same damaged copy take turns and read one published copy")
    func serializesRepair() async throws {
        let fixture = try BundleFixture()
        defer { fixture.remove() }
        _ = try ValidatedSpeechBundle.open(fixture.source, copyingInto: fixture.copies, layout: BundleFixture.layout)
        try FileManager.default.removeItem(at: fixture.copy.appending(path: "artifacts/a.aimodel/main.mlirb"))
        let (source, lockPath) = (fixture.source, fixture.copies.appending(path: ".lock").path)
        let laterReads = LifetimeLog()
        let laterRuntime = IrodoriRuntime(dependencies: fixture.dependencies(TestDriver(), reads: laterReads))
        let later = Mutex<Task<SpeechModel, any Error>?>(nil)
        let lockStates = LifetimeLog()
        let startLater: @Sendable (Int32, String) -> Int32 = { descriptor, path in
            if path.hasSuffix("/irodori.json") {
                let probe = Darwin.open(lockPath, O_RDWR | O_CREAT | O_CLOEXEC, 0o600)
                let held = probe >= 0 && flock(probe, LOCK_EX | LOCK_NB) != 0 && errno == EWOULDBLOCK
                lockStates.add(held ? "held" : "free")
                close(probe)
                later.withLock { $0 = Task { try await laterRuntime.open(source) } }
            }
            return ValidatedSpeechBundle.cloneFile(descriptor, to: path)
        }
        let reads = LifetimeLog()
        let runtime = IrodoriRuntime(dependencies: fixture.dependencies(TestDriver(), reads: reads, clone: startLater))
        _ = try await runtime.open(source)
        let published = try fixture.inodes()
        _ = try await #require(later.withLock { $0 }).value
        _ = try await runtime.generate(text: "13", seed: 1)
        _ = try await laterRuntime.generate(text: "13", seed: 1)
        #expect(lockStates.snapshot() == ["held"])
        #expect(try fixture.inodes() == published)
        #expect(try fixture.copyFolders() == [BundleFixture.key])
        let tokenizer = "\(fixture.copy.appending(path: "tokenizer.json").path)=tokenizer-1"
        let assets = ["a", "b"].map { "\(fixture.copy.appending(path: "artifacts/\($0).aimodel").path)=model-\($0)-1" }
        #expect(reads.snapshot() == [tokenizer] + assets)
        #expect(laterReads.snapshot() == [tokenizer] + assets)
    }

    @Test("SPK-04 After a clone failure the iOS setting rejects the folder and the Mac setting verifies it in place")
    func cloneFailure() async throws {
        let fixture = try BundleFixture()
        defer { fixture.remove() }
        let manifestOnly: @Sendable (Int32, String) -> Int32 = {
            $1.hasSuffix("/irodori.json") ? ValidatedSpeechBundle.cloneFile($0, to: $1) : EXDEV
        }
        let phoneReads = LifetimeLog()
        let phone = IrodoriRuntime(
            dependencies: fixture.dependencies(TestDriver(), reads: phoneReads, inPlace: false, clone: manifestOnly))
        let message = "The model folder could not be copied for verification (error \(EXDEV))."
        await #expect(throws: SpeechError.invalidAsset(message)) { try await phone.open(fixture.source) }
        #expect(try fixture.copyFolders().isEmpty)
        #expect(phoneReads.snapshot().isEmpty)
        await #expect(throws: SpeechError.noModel) { try await phone.generate(text: "13", seed: 1) }
        let macReads = LifetimeLog()
        let mac = IrodoriRuntime(
            dependencies: fixture.dependencies(TestDriver(), reads: macReads, inPlace: true, clone: manifestOnly))
        _ = try await mac.open(fixture.source)
        _ = try await mac.generate(text: "13", seed: 1)
        #expect(try fixture.copyFolders().isEmpty)
        let source = fixture.source
        #expect(
            macReads.snapshot() == [
                "\(source.appending(path: "tokenizer.json").path)=tokenizer-1",
                "\(source.appending(path: "artifacts/a.aimodel").path)=model-a-1",
                "\(source.appending(path: "artifacts/b.aimodel").path)=model-b-1",
            ])
        try fixture.apply(.changedBytes)
        await #expect(throws: SpeechError.self) { try await mac.open(fixture.source) }
        #expect(ValidatedSpeechBundle.inPlaceWhenCloneFails)
        let root = ValidatedSpeechBundle.defaultRoot
        #expect(root.lastPathComponent == "IrodoriVerified")
        #expect(root.deletingLastPathComponent().deletingLastPathComponent().path == URL.cachesDirectory.path)
    }
}

enum SourceDefect: CaseIterable, Sendable {
    case symlinkedFile, symlinkedFolder, fifo, fifoManifest, changedBytes, missingFile
}

struct BundleFixture: Sendable {
    static let original = [
        "irodori.json": "manifest",
        "tokenizer.json": "tokenizer-1",
        "artifacts/a.aimodel/main.mlirb": "model-a-1",
        "artifacts/b.aimodel/main.mlirb": "model-b-1",
    ]
    static let replacements = [
        "tokenizer.json": "tokenizer-2",
        "artifacts/a.aimodel/main.mlirb": "model-a-2",
        "artifacts/b.aimodel/main.mlirb": "model-b-2",
    ]
    static let key = digest("manifest")

    let base = FileManager.default.temporaryDirectory.appending(
        path: "VerifiedCopyTests-" + UUID().uuidString, directoryHint: .isDirectory)
    var source: URL { base.appending(path: "source", directoryHint: .isDirectory) }
    var copies: URL { base.appending(path: "copies", directoryHint: .isDirectory) }
    var copy: URL { copies.appending(path: Self.key, directoryHint: .isDirectory) }

    init() throws {
        for (path, text) in Self.original { try write(text, to: path) }
    }

    static func digest(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    static func layout(_ manifest: Data) throws -> SpeechBundleLayout {
        guard manifest == Data("manifest".utf8) else { throw SpeechError.invalidAsset("Unknown test manifest") }
        return SpeechBundleLayout(
            files: original.map { path, text in
                IrodoriBundleInventory.File(path: path, bytes: Int64(text.utf8.count), sha256: digest(text))
            }, assets: ["artifacts/a.aimodel", "artifacts/b.aimodel"])
    }

    func dependencies(
        _ driver: TestDriver, reads: LifetimeLog, inPlace: Bool = false,
        clone: @escaping @Sendable (Int32, String) -> Int32 = ValidatedSpeechBundle.cloneFile(_:to:)
    ) -> SpeechDependencies {
        let copies = copies
        return SpeechDependencies(
            validate: {
                try ValidatedSpeechBundle.open(
                    $0, copyingInto: copies, layout: Self.layout, inPlaceWhenCloneFails: inPlace, clone: clone)
            },
            tokenizer: { url in
                reads.add("\(url.path)=\(try String(contentsOf: url, encoding: .utf8))")
                return TestTokenizer()
            },
            pipeline: { assets in
                for asset in assets {
                    let model = try String(contentsOf: asset.appending(path: "main.mlirb"), encoding: .utf8)
                    reads.add("\(asset.path)=\(model)")
                }
                return await driver.makePipeline()
            })
    }

    func write(_ text: String, to path: String, in folder: URL? = nil) throws {
        let url = (folder ?? source).appending(path: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }

    func overwrite(_ path: String, with text: String) throws {
        let handle = try FileHandle(forWritingTo: source.appending(path: path))
        try handle.truncate(atOffset: 0)
        try handle.write(contentsOf: Data(text.utf8))
        try handle.close()
    }

    func duplicateSource() throws -> URL {
        let duplicate = base.appending(path: "duplicate", directoryHint: .isDirectory)
        for (path, text) in Self.original { try write(text, to: path, in: duplicate) }
        return duplicate
    }

    func apply(_ defect: SourceDefect) throws {
        let files = FileManager.default
        switch defect {
        case .symlinkedFile:
            let outside = base.appending(path: "outside-tokenizer.json")
            try files.moveItem(at: source.appending(path: "tokenizer.json"), to: outside)
            try files.createSymbolicLink(at: source.appending(path: "tokenizer.json"), withDestinationURL: outside)
        case .symlinkedFolder:
            let outside = base.appending(path: "outside.aimodel", directoryHint: .isDirectory)
            try files.moveItem(at: source.appending(path: "artifacts/a.aimodel"), to: outside)
            try files.createSymbolicLink(at: source.appending(path: "artifacts/a.aimodel"), withDestinationURL: outside)
        case .fifo, .fifoManifest:
            let path = source.appending(path: defect == .fifo ? "artifacts/b.aimodel/main.mlirb" : "irodori.json").path
            try files.removeItem(atPath: path)
            try #require(mkfifo(path, 0o600) == 0)
        case .changedBytes:
            try overwrite("tokenizer.json", with: "tokenizer-X")
        case .missingFile:
            try files.removeItem(at: source.appending(path: "artifacts/a.aimodel/main.mlirb"))
        }
    }

    func copyFolders() throws -> Set<String> {
        guard FileManager.default.fileExists(atPath: copies.path) else { return [] }
        return Set(try FileManager.default.contentsOfDirectory(atPath: copies.path)).subtracting([".lock"])
    }

    func inodes() throws -> [String: UInt64] {
        var result: [String: UInt64] = [:]
        for path in try FileManager.default.subpathsOfDirectory(atPath: copy.path) {
            var status = stat()
            if lstat(copy.appending(path: path).path, &status) == 0, status.st_mode & S_IFMT == S_IFREG {
                result[path] = status.st_ino
            }
        }
        return result
    }

    func remove() { try? FileManager.default.removeItem(at: base) }
}
