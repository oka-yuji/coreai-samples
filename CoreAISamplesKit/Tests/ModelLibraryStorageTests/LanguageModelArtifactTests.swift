import Foundation
import ModelLibraryFeature
import Testing

@testable import ModelLibraryStorage

struct LanguageModelArtifactTests {
    @Test("LLM-01 The Mac language artifacts pin the published repositories, revisions and complete inventories")
    func publishedInventories() throws {
        let catalog = try MacDemoCatalog.make()
        let qwen = try #require(catalog.model("qwen3.8-27b")?.artifacts.first)
        let clef = try #require(catalog.model("clef-flash-9b")?.artifacts.first)
        #expect(qwen.download?.repository == "okayuji/Qwen3.8-27B-coreai-4bit")
        #expect(qwen.download?.revision == "8e24aebe036eae577d4a3bd019ba728a73358318")
        #expect(clef.download?.repository == "okayuji/clef-flash-9b-coreai-decide")
        #expect(clef.download?.revision == "12e30be4e391310377c534524df8962359a08b0f")
        #expect(qwen.files.count == 12 && qwen.totalFileBytes == 16_852_543_780)
        #expect(clef.files.count == 25 && clef.totalFileBytes == 7_278_592_396)
        for artifact in [qwen, clef] {
            #expect(artifact.isSourceValid && artifact.platforms == [.macOS])
            #expect(Set(artifact.files.map(\.path)).count == artifact.files.count)
        }
    }

    @Test(
        "LLM-02 Language and decision families accept the pinned manifest and reject an altered manifest",
        arguments: [false, true])
    func manifests(decision: Bool) async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let sample = try sample(decision: decision)
        let store = fixture.store()
        let resource = try await fixture.install(sample, into: store)
        #expect(try await store.inspect(model: sample.model, artifact: sample.artifact).resource == resource)
        var changed = sample.files
        let path = decision ? "decision.json" : "metadata.json"
        changed[path]?.append(Data("\n".utf8))
        let altered = sample.with(files: changed)
        await #expect(throws: ManagedModelStoreError.contractMismatch(path)) {
            try await fixture.install(altered, into: store)
        }
        #expect(try await store.inspect(model: sample.model, artifact: sample.artifact).resource == resource)
    }

    @Test("LLM-03 A complete legacy Qwen folder migrates and is removed only after the managed copy is recorded")
    func movesCompleteFolder() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let sample = try sample()
        let source = fixture.documents.appending(path: "legacy-qwen")
        try fixture.write(
            sample.files.merging([".download-complete.json": Data("{}".utf8)]) { first, _ in first }, to: source)
        let store = ManagedModelStore(root: fixture.root, removeMigratedSources: true)
        let report = await store.migrate(candidates: [source], model: sample.model, artifact: sample.artifact)
        #expect(report.outcomes[source]?.resource != nil)
        #expect(!FileManager.default.fileExists(atPath: source.path))
        #expect(try await store.inspect(model: sample.model, artifact: sample.artifact).resource != nil)
    }

    @Test(
        "LLM-03 Incomplete folders, insufficient space and unknown source files keep the original folder",
        arguments: ["missing", "space", "extra"])
    func preservesRejectedSources(_ defect: String) async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let sample = try sample()
        let source = fixture.documents.appending(path: "legacy-qwen")
        var files = sample.files
        if defect == "missing" { files["payload.bin"] = nil }
        if defect == "extra" { files["my-notes.txt"] = Data("keep".utf8) }
        try fixture.write(files, to: source)
        let before = fixture.contents(source)
        let store = ManagedModelStore(
            root: fixture.root, removeMigratedSources: true, availableCapacity: { _ in defect == "space" ? 0 : .max })
        let report = await store.migrate(candidates: [source], model: sample.model, artifact: sample.artifact)
        #expect(report.outcomes[source]?.resource == nil)
        #expect(fixture.contents(source) == before)
        #expect(try await store.inspect(model: sample.model, artifact: sample.artifact).isMissing)
    }

    @Test("LLM-03 Cleanup resumes after cancellation following promotion without copying the model again")
    func resumesCleanup() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let sample = try sample()
        let source = fixture.documents.appending(path: "legacy-qwen")
        try fixture.write(sample.files, to: source)
        let store = ManagedModelStore(
            root: fixture.root, removeMigratedSources: true,
            writeLedger: {
                try $0.write(to: $1, options: .atomic)
                withUnsafeCurrentTask { $0?.cancel() }
            })
        let first = await Task {
            await store.migrate(candidates: [source], model: sample.model, artifact: sample.artifact)
        }.value
        #expect(first.outcomes[source]?.failure == describe(CancellationError()))
        #expect(fixture.contents(source) == sample.files.mapValues(sha256))
        let ledger = try fixture.ledger()
        let next = ManagedModelStore(root: fixture.root, removeMigratedSources: true, availableCapacity: { _ in 0 })
        let result = await next.migrate(candidates: [source], model: sample.model, artifact: sample.artifact)
        #expect(result.outcomes[source]?.isSkipped == true)
        #expect(!FileManager.default.fileExists(atPath: source.path))
        #expect(try fixture.ledger().entries.map(\.generation) == ledger.entries.map(\.generation))
    }

    @Test("Runtime leases reject uninstalled and mismatched paths and keep deletion blocked until release")
    func runtimeLease() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let sample = try sample()
        let store = fixture.store()
        await #expect(throws: ManagedModelStoreError.notInstalled) {
            try await store.acquire(
                model: sample.model, artifact: sample.artifact,
                directory: fixture.artifactDirectory(sample), owner: UUID())
        }
        let resource = try await fixture.install(sample, into: store)
        await #expect(throws: ManagedModelStoreError.notInstalled) {
            try await store.acquire(
                model: sample.model, artifact: sample.artifact,
                directory: fixture.documents, owner: UUID())
        }
        let first = try await store.acquire(
            model: sample.model, artifact: sample.artifact,
            directory: URL(filePath: resource.directory.path), owner: UUID())
        let second = try await store.acquire(
            model: sample.model, artifact: sample.artifact,
            directory: URL(filePath: resource.directory.path), owner: UUID())
        await first.release()
        await #expect(throws: ManagedModelStoreError.inUse) { try await store.delete(identity: resource.identity) }
        await second.release()
        #expect(try await store.delete(identity: resource.identity) > 0)
    }

    @Test("Runtime use counts preserve another window's protection until every holder releases")
    func runtimeUseCounts() {
        var state = ModelRuntimeUses.State()
        #expect(ModelRuntimeUses.reduce(&state, .acquired("qwen")).isEmpty)
        _ = ModelRuntimeUses.reduce(&state, .acquired("qwen"))
        _ = ModelRuntimeUses.reduce(&state, .acquired("clef"))
        _ = ModelRuntimeUses.reduce(&state, .released("qwen"))
        #expect(state.counts == ["qwen": 1, "clef": 1])
        _ = ModelRuntimeUses.reduce(&state, .released("qwen"))
        _ = ModelRuntimeUses.reduce(&state, .released("qwen"))
        #expect(state.counts == ["clef": 1])
    }

    @Test("LLM-04 Only the exact legacy location matches the managed model identity")
    func conversationLocations() {
        let legacy = URL(filePath: "/test/models/owner/qwen")
        let managed = URL(filePath: "/test/library/artifacts/qwen/revision")
        let key = { ModelLocationIdentity.key($0, legacy: legacy, managed: managed) }
        #expect(key(legacy) == key(managed))
        #expect(key(legacy.appending(path: ".")) == key(managed))
        #expect(key(URL(filePath: "/test/other/qwen")) != key(managed))
        #expect(key(legacy.appending(path: "other")) != key(managed))
    }

    private func sample(decision: Bool = false) throws -> Sample {
        let key = decision ? "clef" : "qwen"
        let url = try #require(
            Bundle.module.url(forResource: key + "-manifest", withExtension: "json", subdirectory: "Resources"))
        return Sample(
            key, model: key,
            contract: (decision ? LanguageModelArtifacts.clefID : LanguageModelArtifacts.qwenID) + "-v1",
            files: [
                decision ? "decision.json" : "metadata.json": try Data(contentsOf: url),
                "payload.bin": Data("weights".utf8),
            ],
            source: nil)
    }
}
