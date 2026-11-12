import Foundation
import ModelLibraryFeature
import Testing

@testable import ModelLibraryStorage

struct ChronosImportTests {
    @Test("The pinned Chronos bundle is recognized by the managed store before importing")
    func definition() async throws {
        let artifact = try ChronosModelArtifact.make()
        #expect(artifact.isSourceValid && artifact.sourceKind == .localManifest && artifact.download?.isValid == true)
        #expect(artifact.files.count == 9 && artifact.totalFileBytes == 478_864_956)
        #expect(artifact.sizeBytes == artifact.totalFileBytes && artifact.platforms == [.macOS, .iOS])
        #expect(
            artifact.upstream == ModelUpstream(repository: "amazon/chronos-2", revision: ChronosModelArtifact.revision))
        #expect(artifact.sourceRevision == ChronosModelArtifact.manifestSHA256)
        #expect(try JSONDecoder().decode(ModelArtifact.self, from: JSONEncoder().encode(artifact)) == artifact)
        let model = try #require(MacDemoCatalog.make().model("chronos2"))
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store(availableCapacity: { _ in 1_000_000_000 })
        #expect(try await store.inspect(model: model, artifact: artifact).isMissing)
        #expect(
            try await store.stagingDirectory(for: model, artifact: artifact).lastPathComponent
                == ChronosModelArtifact.manifestSHA256)
    }

    @Test(
        "Chronos metadata is trusted only when every original manifest byte matches the pinned contract",
        arguments: ["format", "precision", "source", "abi", "quantiles", "raw_context", "files", "extra", "encoding"])
    func metadata(_ field: String) throws {
        let fixture = try ChronosImportFixture()
        let raw = try fixture.manifest()
        #expect(sha256(raw) == ChronosModelArtifact.manifestSHA256)
        try ChronosModelArtifact.validateManifest(raw)
        var value = try #require(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        if field != "encoding" { value[field] = "changed" }
        #expect(throws: ManagedModelStoreError.contractMismatch("Chronos metadata")) {
            try ChronosModelArtifact.validateManifest(JSONSerialization.data(withJSONObject: value))
        }
        let files = try #require(JSONSerialization.jsonObject(with: raw) as? [String: Any])["files"]
        let inventory = try #require(files as? [String: [String: Any]])
        #expect(Set(inventory.keys) == Set(fixture.artifact.files.map(\.path)).subtracting(["chronos2.json"]))
        for file in fixture.artifact.files where file.path != "chronos2.json" {
            #expect(inventory[file.path]?["sha256"] as? String == file.sha256)
            #expect((inventory[file.path]?["bytes"] as? NSNumber)?.int64Value == file.bytes)
        }
    }

    @Test(
        "A forged Chronos definition cannot replace the trusted inventory or metadata identity",
        arguments: [
            "hash", "bytes", "missing", "extra", "source", "manifest", "profile", "contract", "platforms", "download",
        ])
    func forgedDefinition(_ field: String) async throws {
        let fixture = try ChronosImportFixture()
        defer { fixture.files.remove() }
        let raw = try JSONEncoder().encode(fixture.artifact)
        var value = try #require(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        var files = try #require(value["files"] as? [[String: Any]])
        switch field {
        case "hash": files[0]["sha256"] = String(repeating: "a", count: 64)
        case "bytes": files[0]["bytes"] = 1
        case "missing": files.removeFirst()
        case "extra": files.append(["path": "extra", "bytes": 0, "sha256": sha256(Data())])
        case "source": value["upstream"] = ["repository": "other/model", "revision": ChronosModelArtifact.revision]
        case "manifest":
            let changed = String(repeating: "b", count: 64)
            value["localBundle"] = [
                "manifestSHA256": changed, "profileID": "chronos2-fp32", "manifestFile": "chronos2.json",
            ]
            let index = try #require(files.firstIndex { $0["path"] as? String == "chronos2.json" })
            files[index]["sha256"] = changed
        case "profile":
            value["localBundle"] = [
                "manifestSHA256": ChronosModelArtifact.manifestSHA256, "profileID": "other",
                "manifestFile": "chronos2.json",
            ]
        case "contract": value["contract"] = "chronos2-other"
        case "platforms": value["platforms"] = ["macOS"]
        default:
            value.removeValue(forKey: "localBundle")
            value["download"] = ["repository": "other/model", "revision": ChronosModelArtifact.revision, "root": ""]
        }
        value["files"] = files
        let artifact = try JSONDecoder().decode(ModelArtifact.self, from: JSONSerialization.data(withJSONObject: value))
        #expect(artifact.isSourceValid)
        let model = LibraryModel(
            id: "chronos2", name: "Chronos-2", category: "Time Series Forecasting", artifacts: [artifact],
            defaultArtifactID: artifact.id, platforms: [.macOS], demos: [])
        let store = fixture.store()
        #expect(try await store.inspect(model: model, artifact: artifact).reason != nil)
        await #expect(throws: (any Error).self) {
            try await store.stagingDirectory(for: model, artifact: artifact)
        }
    }

    @Test("The pinned Chronos manifest reaches payload hashing before synthetic content is rejected")
    func manifestReachesPayload() async throws {
        let fixture = try ChronosImportFixture()
        defer { fixture.files.remove() }
        try fixture.sparseSource(at: fixture.files.documents)
        let before = try fixture.sourceSummary(fixture.files.documents)
        let store = fixture.store()
        await #expect(throws: ManagedModelStoreError.checksumMismatch("LICENSE")) {
            try await store.copyForImport(
                from: fixture.files.documents, model: fixture.model, artifact: fixture.artifact
            ) { _ in }
        }
        #expect(try await store.inspect(model: fixture.model, artifact: fixture.artifact).isMissing)
        #expect(try fixture.sourceSummary(fixture.files.documents) == before)
    }

    @Test(
        "Rejected Chronos replacements preserve the recorded synthetic installation and the input folder",
        arguments: ["altered", "manifest", "missing", "extra", "symlink"])
    func rejectedReplacement(_ kind: String) async throws {
        let fixture = try ChronosImportFixture()
        defer { fixture.files.remove() }
        let installed = try fixture.recordSyntheticInstallation()
        let store = fixture.store()
        #expect(try await store.inspect(model: fixture.model, artifact: fixture.artifact).resource == installed)
        let ledger = try Data(contentsOf: fixture.files.root.appending(path: "ledger.json"))
        let previous = try fixture.sourceSummary(installed.directory)
        try fixture.sparseSource(at: fixture.files.documents)
        let expected = try fixture.damage(kind, at: fixture.files.documents)
        let source = try fixture.sourceSummary(fixture.files.documents)
        await #expect(throws: expected) {
            try await store.copyForImport(
                from: fixture.files.documents, model: fixture.model, artifact: fixture.artifact
            ) { _ in }
        }
        #expect(try await store.inspect(model: fixture.model, artifact: fixture.artifact).resource == installed)
        #expect(try Data(contentsOf: fixture.files.root.appending(path: "ledger.json")) == ledger)
        #expect(try fixture.sourceSummary(installed.directory) == previous)
        #expect(try fixture.sourceSummary(fixture.files.documents) == source)
        let staging = try await store.stagingDirectory(for: fixture.model, artifact: fixture.artifact)
        #expect(try FileManager.default.contentsOfDirectory(atPath: staging.path).isEmpty)
    }

    @Test("Chronos promotion and full inspection reject synthetic payloads despite an existing ready ledger")
    func fullVerification() async throws {
        let fixture = try ChronosImportFixture()
        defer { fixture.files.remove() }
        let installed = try fixture.recordSyntheticInstallation()
        let store = fixture.store()
        let staging = try await store.stagingDirectory(for: fixture.model, artifact: fixture.artifact)
        try fixture.sparseSource(at: staging)
        await #expect(throws: ManagedModelStoreError.checksumMismatch("LICENSE")) {
            try await store.promote(staging: staging, model: fixture.model, artifact: fixture.artifact)
        }
        let checked = try await store.inspect(model: fixture.model, artifact: fixture.artifact, verification: .full)
        #expect(checked.reason == describe(ManagedModelStoreError.checksumMismatch("LICENSE")))
        #expect(try await store.inspect(model: fixture.model, artifact: fixture.artifact).resource == installed)
    }

    @Test("The last Chronos lease must be released before deleting a recorded synthetic installation")
    func leaseGating() async throws {
        let fixture = try ChronosImportFixture()
        defer { fixture.files.remove() }
        let installed = try fixture.recordSyntheticInstallation()
        let store = fixture.store()
        let first = try await store.acquire(
            model: fixture.model, artifact: fixture.artifact, directory: installed.directory, owner: UUID())
        let second = try await store.acquire(
            model: fixture.model, artifact: fixture.artifact, directory: installed.directory, owner: UUID())
        await #expect(throws: ManagedModelStoreError.inUse) { try await store.delete(identity: installed.identity) }
        await #expect(throws: ManagedModelStoreError.inUse) {
            try await store.copyForImport(
                from: fixture.files.documents, model: fixture.model, artifact: fixture.artifact
            ) { _ in }
        }
        await first.release()
        #expect(await store.activeLeases(identity: installed.identity) == 1)
        await #expect(throws: ManagedModelStoreError.inUse) { try await store.delete(identity: installed.identity) }
        await second.release()
        #expect(await store.activeLeases(identity: installed.identity) == 0)
        #expect(try await store.delete(identity: installed.identity) == installed.sizeBytes)
        #expect(try await store.inspect(model: fixture.model, artifact: fixture.artifact).isMissing)
    }

    @Test(
        "The pinned Chronos bundle imports intact, preserves its source and old registration and respects leases",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_CHRONOS2_IMPORT_BUNDLE"] != nil))
    func knownBundle() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_CHRONOS2_IMPORT_BUNDLE"])
        let source = URL(filePath: path, directoryHint: .isDirectory)
        let fixture = try ChronosImportFixture()
        defer { fixture.files.remove() }
        let before = try fixture.hashes(source)
        #expect(before == Dictionary(uniqueKeysWithValues: fixture.artifact.files.map { ($0.path, $0.sha256) }))
        let store = fixture.store()
        let staging = try await store.copyForImport(
            from: source, model: fixture.model, artifact: fixture.artifact
        ) { _ in }
        let installed = try await store.promote(staging: staging, model: fixture.model, artifact: fixture.artifact)
        #expect(installed.matches(model: fixture.model, artifact: fixture.artifact))
        #expect(installed.sourceKind == .localManifest && installed.directory != source)
        #expect(installed.revision == ChronosModelArtifact.manifestSHA256 && installed.sizeBytes == 478_864_956)
        #expect(
            try await store.inspect(model: fixture.model, artifact: fixture.artifact, verification: .full).resource
                == installed)
        let lease = try await store.acquire(
            model: fixture.model, artifact: fixture.artifact, directory: installed.directory, owner: UUID())
        await #expect(throws: ManagedModelStoreError.inUse) { try await store.delete(identity: installed.identity) }
        await lease.release()
        let ledger = try Data(contentsOf: fixture.files.root.appending(path: "ledger.json"))
        for kind in ["altered", "manifest", "missing", "extra", "symlink"] {
            try FileManager.default.copyItem(at: source, to: fixture.files.documents)
            let expected = try fixture.damage(kind, at: fixture.files.documents)
            let damaged = try fixture.sourceSummary(fixture.files.documents)
            await #expect(throws: expected) {
                try await store.copyForImport(
                    from: fixture.files.documents, model: fixture.model, artifact: fixture.artifact
                ) { _ in }
            }
            #expect(try await store.inspect(model: fixture.model, artifact: fixture.artifact).resource == installed)
            #expect(try Data(contentsOf: fixture.files.root.appending(path: "ledger.json")) == ledger)
            #expect(try fixture.sourceSummary(fixture.files.documents) == damaged)
            try FileManager.default.removeItem(at: fixture.files.documents)
        }
        #expect(
            try await store.inspect(model: fixture.model, artifact: fixture.artifact, verification: .full).resource
                == installed)
        #expect(try await store.delete(identity: installed.identity) == installed.sizeBytes)
        #expect(try await store.inspect(model: fixture.model, artifact: fixture.artifact).isMissing)
        #expect(try fixture.hashes(source) == before)
    }
}
