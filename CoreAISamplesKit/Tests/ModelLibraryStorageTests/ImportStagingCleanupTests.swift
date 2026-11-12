import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTestSupport
import Testing

struct ImportStagingCleanupTests {
    @Test("CD-02 Successful import removes obsolete download files while preserving other staging owners")
    func successfulImport() async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let area = try await store.stagingDirectory(for: fixture.model, artifact: fixture.artifact)
        let nested = try #require(fixture.files.keys.sorted().first { $0.hasSuffix("weights.bin") })
        let obsolete = ["LICENSE", "NOTICE.txt.part", nested, nested + ".part"]
        for path in obsolete { try write(Data("download bytes".utf8), to: area.appending(path: path)) }
        let unknown = area.appending(path: "unrelated.txt")
        let otherArtifact = fixture.root.appending(path: "staging/another-artifact/revision/LICENSE")
        for url in [unknown, otherArtifact] { try write(Data("keep".utf8), to: url) }
        let pending = try await store.copyForImport(
            from: fixture.source, model: fixture.model, artifact: fixture.artifact, progress: { _ in })
        let importing = try await store.copyForImport(
            from: fixture.source, model: fixture.model, artifact: fixture.artifact, progress: { _ in })
        let installed = try await store.promote(staging: importing, model: fixture.model, artifact: fixture.artifact)
        let remaining = obsolete.filter { FileManager.default.fileExists(atPath: area.appending(path: $0).path) }
        #expect(remaining.isEmpty)
        for url in [unknown, otherArtifact] { #expect(try Data(contentsOf: url) == Data("keep".utf8)) }
        for (path, data) in fixture.files {
            #expect(try Data(contentsOf: fixture.source.appending(path: path)) == data)
            #expect(try Data(contentsOf: pending.appending(path: path)) == data)
            #expect(try Data(contentsOf: installed.directory.appending(path: path)) == data)
        }
        #expect(await store.activeLeases(identity: installed.identity) == 0)
    }

    @Test(
        "CD-02 Failed imports retain resumable download files and the previously ready installation",
        arguments: ["copy", "cancel", "capacity", "ledger", "lease"])
    func failedImport(_ failure: String) async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let installed = try await fixture.install(into: store)
        let ledger = fixture.root.appending(path: "ledger.json")
        let recorded = try Data(contentsOf: ledger)
        let area = try await store.stagingDirectory(for: fixture.model, artifact: fixture.artifact)
        let bytes = Data("download bytes".utf8)
        let paths = ["LICENSE", "NOTICE.txt.part"]
        for path in paths { try write(bytes, to: area.appending(path: path)) }
        switch failure {
        case "copy":
            try Data("changed".utf8).write(to: fixture.source.appending(path: "LICENSE"))
            await #expect(throws: ManagedModelStoreError.sizeMismatch("LICENSE")) {
                try await store.copyForImport(
                    from: fixture.source, model: fixture.model, artifact: fixture.artifact, progress: { _ in })
            }
        case "cancel":
            let operation = Task {
                try await store.copyForImport(
                    from: fixture.source, model: fixture.model, artifact: fixture.artifact
                ) { _ in withUnsafeCurrentTask { $0?.cancel() } }
            }
            await #expect(throws: CancellationError.self) { try await operation.value }
        case "capacity":
            await #expect(throws: ManagedModelStoreError.insufficientSpace) {
                try await fixture.store(available: 0).copyForImport(
                    from: fixture.source, model: fixture.model, artifact: fixture.artifact, progress: { _ in })
            }
        case "ledger":
            let failing = ManagedModelStore(
                root: fixture.root, writeLedger: { _, _ in throw CocoaError(.fileWriteOutOfSpace) })
            let staging = try await failing.copyForImport(
                from: fixture.source, model: fixture.model, artifact: fixture.artifact, progress: { _ in })
            await #expect(throws: CocoaError.self) {
                try await failing.promote(staging: staging, model: fixture.model, artifact: fixture.artifact)
            }
            try await failing.discardImport(staging: staging, model: fixture.model, artifact: fixture.artifact)
        default:
            let staging = try await store.copyForImport(
                from: fixture.source, model: fixture.model, artifact: fixture.artifact, progress: { _ in })
            let lease = await store.lease(identity: installed.identity, owner: UUID())
            await #expect(throws: ManagedModelStoreError.inUse) {
                try await store.promote(staging: staging, model: fixture.model, artifact: fixture.artifact)
            }
            #expect(await store.activeLeases(identity: installed.identity) == 1)
            await lease.release()
            try await store.discardImport(staging: staging, model: fixture.model, artifact: fixture.artifact)
        }
        for path in paths { #expect(try Data(contentsOf: area.appending(path: path)) == bytes) }
        #expect(try Data(contentsOf: ledger) == recorded)
        guard
            case .ready(let resource) = try await store.inspect(
                model: fixture.model, artifact: fixture.artifact, verification: .full)
        else {
            Issue.record("The failed import lost its previous ready resource")
            return
        }
        #expect(resource == installed)
    }

    @Test("CD-02 Successful import never follows staging symlinks", arguments: ["parent", "file", "part"])
    func symbolicLinks(_ target: String) async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let area = try await store.stagingDirectory(for: fixture.model, artifact: fixture.artifact)
        let nested = try #require(fixture.files.keys.sorted().first { $0.hasSuffix("weights.bin") })
        let parts = nested.split(separator: "/")
        let root = fixture.root.appending(path: "staging/another-artifact/revision")
        let sentinel = root.appending(path: target == "parent" ? parts.dropFirst().joined(separator: "/") : "sentinel")
        let bytes = Data("another owner".utf8)
        try write(bytes, to: sentinel)
        let link = area.appending(
            path: target == "parent" ? String(parts[0]) : target == "file" ? "LICENSE" : "NOTICE.txt.part")
        let destination = target == "parent" ? root : sentinel
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: destination)
        _ = try await fixture.install(into: store)
        #expect(try Data(contentsOf: sentinel) == bytes)
        #expect(try FileManager.default.destinationOfSymbolicLink(atPath: link.path) == destination.path)
    }

    @Test("CD-02 A UUID staging directory remains reserved even if an inventory path shares its prefix")
    func reservedImportDirectory() async throws {
        let owner = UUID().uuidString
        let path = owner + "/reserved.bin"
        let fixture = try ImageBundleFixture { _, files in files[path] = Data("fixture".utf8) }
        defer { fixture.remove() }
        let store = fixture.store()
        let area = try await store.stagingDirectory(for: fixture.model, artifact: fixture.artifact)
        let sentinel = area.appending(path: path)
        let bytes = Data("pending import".utf8)
        try write(bytes, to: sentinel)
        _ = try await fixture.install(into: store)
        #expect(try Data(contentsOf: sentinel) == bytes)
    }

    private func write(_ data: Data, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }
}
