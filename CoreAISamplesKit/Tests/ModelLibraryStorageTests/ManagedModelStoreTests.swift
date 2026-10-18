import Foundation
import ModelLibraryFeature
import Testing

@testable import ModelLibraryStorage

struct ManagedModelStoreTests {
    @Test("STORE-01 Intact GLiNER and Embedding fixtures are the only bundles returned as validated resources")
    func acceptsIntactFixtures() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let unconfigured = ModelArtifact(id: "plain", title: "Plain", contract: "unconfigured", platforms: [.iOS])
        let plain = LibraryModel(
            id: "plain", name: "Plain", category: "Fixture", artifacts: [unconfigured], defaultArtifactID: "plain",
            platforms: [.iOS], demos: [])
        #expect(try await store.inspect(model: plain, artifact: unconfigured).isMissing)
        for (index, sample) in try [Sample.gliner(), Sample.embedding(), Sample.embedding(int4: true)].enumerated() {
            #expect(try await store.inspect(sample).isMissing)
            let promoted = try await fixture.install(sample, into: store)
            #expect(try await store.inspect(sample).resource == promoted)
            #expect(promoted.matches(model: sample.model, artifact: sample.artifact))
            #expect(
                promoted.revision == distribution && promoted.upstreamRevision == sample.artifact.upstream?.revision)
            #expect(promoted.directory.path == fixture.artifactDirectory(sample).path)
            #expect(promoted.sizeBytes == sample.artifact.sizeBytes)
            #expect(
                Set(fixture.contents(promoted.directory).filter { $0.value != "directory" }.keys)
                    == Set(sample.files.keys))
            let entry = try #require(try fixture.ledger().entries.first { $0.artifactID == sample.artifact.id })
            #expect(entry.generation == index + 1 && entry.installedAt == Date(timeIntervalSince1970: 1_000))
            #expect(await store.generation(of: promoted.identity) == index + 1)
        }
        let gliner = try Sample.gliner()
        let embedding = try Sample.embedding()
        #expect(
            try await store.inspect(model: embedding.model, artifact: gliner.artifact).reason
                == describe(ManagedModelStoreError.invalidDefinition))
        let unpinned = try Sample.gliner(revision: "main")
        try fixture.write(unpinned.files, to: fixture.artifactDirectory(unpinned))
        #expect(try await store.inspect(unpinned).isMissing)
        let staging = fixture.stagingArea(unpinned)
        try fixture.write(unpinned.files, to: staging)
        await #expect(throws: ManagedModelStoreError.invalidDefinition) { try await store.promote(staging, unpinned) }
    }

    @Test(
        "STORE-01 Identity, profile, tokenizer, hash and path defects are rejected with their reason",
        arguments: Defect.allCases)
    func rejectsDefects(_ defect: Defect) async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let (sample, files, link, expected) = try defect.make()
        let staging = fixture.stagingArea(sample)
        try fixture.place(files, in: staging, linking: link)
        let refused = await #expect(throws: (any Error).self) { try await store.promote(staging, sample) }
        #expect(describe(refused) == describe(expected))
        #expect(!FileManager.default.fileExists(atPath: fixture.artifactDirectory(sample).path))
        try fixture.place(files, in: fixture.artifactDirectory(sample), linking: link)
        #expect(try await store.inspect(sample).reason == describe(expected))
    }

    @Test("STORE-02 Interrupted, cancelled and space-limited operations never make an incomplete artifact ready")
    func incompleteOperations() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let sample = try Sample.gliner()
        var partial = sample.files
        partial["tokenizer/tokenizer.json"] = nil
        let staging = try await store.staging(sample)
        #expect(staging.path == fixture.stagingArea(sample).path)
        try fixture.write(partial, to: staging)
        await #expect(throws: ManagedModelStoreError.missingFile("tokenizer/tokenizer.json")) {
            try await store.promote(staging, sample)
        }
        #expect(try await store.inspect(sample).isMissing)
        let restarted = fixture.store()
        #expect(try await restarted.staging(sample) == staging)
        #expect(Set(fixture.contents(staging).filter { $0.value != "directory" }.keys) == Set(partial.keys))
        #expect(try await restarted.inspect(sample).isMissing)

        try fixture.write(sample.files, to: staging)
        let staged = fixture.contents(staging)
        let cancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await store.promote(staging, sample)
        }
        await #expect(throws: CancellationError.self) { try await cancelled.value }
        #expect(try await store.inspect(sample).isMissing)
        #expect(fixture.contents(staging) == staged)

        let legacy = fixture.documents.appending(path: "legacy-gliner")
        try fixture.write(sample.files, to: legacy)
        let original = fixture.contents(legacy)
        let limited = await fixture.store(availableCapacity: { _ in 1 }).migrate([legacy], sample)
        #expect(limited.outcomes[legacy]?.failure == describe(ManagedModelStoreError.insufficientSpace))
        #expect(fixture.contents(staging) == staged)
        let interrupting = fixture.store(availableCapacity: { _ in
            withUnsafeCurrentTask { $0?.cancel() }
            return .max
        })
        let interrupted = await Task { await interrupting.migrate([legacy], sample) }.value
        #expect(interrupted.outcomes[legacy]?.failure == describe(CancellationError()))
        #expect(!FileManager.default.fileExists(atPath: staging.path))
        let unrecorded = await fixture.store(writeLedger: { _, _ in throw CocoaError(.fileWriteOutOfSpace) })
            .migrate([legacy], sample)
        #expect(unrecorded.outcomes[legacy]?.failure == describe(CocoaError(.fileWriteOutOfSpace)))
        #expect(!FileManager.default.fileExists(atPath: staging.path))
        #expect(try await store.inspect(sample).isMissing)
        #expect(fixture.contents(legacy) == original)

        try fixture.write(sample.files, to: fixture.artifactDirectory(sample))
        #expect(
            try await fixture.store().inspect(sample).resource?.matches(model: sample.model, artifact: sample.artifact)
                == true)
        #expect(try fixture.ledger().entries.map(\.artifactID) == [sample.artifact.id])
    }

    @Test("STORE-02 Ledger write and version update failures keep the existing intact version")
    func failedUpdatesKeepTheIntactVersion() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let failing = fixture.store(writeLedger: { _, _ in throw CocoaError(.fileWriteOutOfSpace) })
        let current = try Sample.gliner()
        let installed = try await fixture.install(current, into: store)
        let generation = try #require(await store.generation(of: installed.identity))

        let replacement = try await failing.staging(current)
        try fixture.write(current.files, to: replacement)
        await #expect(throws: CocoaError(.fileWriteOutOfSpace)) { try await failing.promote(replacement, current) }
        #expect(try await store.inspect(current).resource == installed)
        #expect(await store.generation(of: installed.identity) == generation)
        #expect(fixture.contents(replacement) == fixture.contents(fixture.artifactDirectory(current)))

        let update = try Sample.gliner(revision: updatedDistribution)
        let unrecorded = try await failing.staging(update)
        try fixture.write(update.files, to: unrecorded)
        await #expect(throws: CocoaError(.fileWriteOutOfSpace)) { try await failing.promote(unrecorded, update) }
        var broken = update.files
        broken["x.aimodel/main.mlirb"] = Data("fixture-gliner-weightZ".utf8)
        let corrupted = try await store.staging(update)
        try fixture.write(broken, to: corrupted)
        await #expect(throws: ManagedModelStoreError.checksumMismatch("x.aimodel/main.mlirb")) {
            try await store.promote(corrupted, update)
        }
        #expect(try await store.inspect(update).isMissing)
        #expect(try await store.inspect(current).resource == installed)

        let lease = await store.lease(identity: installed.identity, owner: UUID())
        let busy = try await store.staging(current)
        try fixture.write(current.files, to: busy)
        await #expect(throws: ManagedModelStoreError.inUse) { try await store.promote(busy, current) }
        #expect(try await store.inspect(current).resource == installed)
        await lease.release()
        #expect(try await store.promote(busy, current) == installed)
        #expect(try #require(await store.generation(of: installed.identity)) > generation)

        let updated = try await fixture.install(update, into: store)
        #expect(try await store.inspect(update).resource == updated)
        #expect(try await store.inspect(current).resource == installed)
    }

    @Test(
        "STORE-03 Known legacy assets migrate non-destructively and repeatably without duplicates, leaving unknown or ambiguous assets and originals unchanged"
    )
    func migratesLegacyAssets() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let gliner = try Sample.gliner()
        let embedding = try Sample.embedding()
        let documents = fixture.documents
        let legacy = documents.appending(path: "gliner-fp32")
        let unknown = documents.appending(path: "gliner-fp16")
        let bundled = documents.appending(path: "gliner-bundled")
        let partial = documents.appending(path: "gliner-partial")
        let linked = documents.appending(path: "gliner-link")
        let aliased = documents.appending(path: "gliner-alias")
        let legacyEmbedding = documents.appending(path: "embeddinggemma2")
        try fixture.write(gliner.files, to: legacy)
        try fixture.write(embedding.files.merging([".DS_Store": Data("finder".utf8)]) { $1 }, to: legacyEmbedding)
        var files = gliner.files
        files["x.aimodel/main.mlirb"] = Data("fixture-gliner-weightZ".utf8)
        try fixture.write(files, to: unknown)
        let unlisted = [
            "SHA256SUMS": Data("sums".utf8), "verification/parity.json": Data("{}".utf8),
            ".DS_Store": Data("finder".utf8), "README.md": Data("readme".utf8),
            "fp16.aimodel/metadata.json": Data("{}".utf8), "._README.md": Data("sidecar".utf8),
            "x.aimodel/._main.mlirb": Data("sidecar".utf8),
        ]
        try fixture.write(gliner.files.merging(unlisted) { $1 }, to: bundled)
        try fixture.write(gliner.files.filter { $0.key != "tokenizer/tokenizer.json" }, to: partial)
        try FileManager.default.createSymbolicLink(at: linked, withDestinationURL: legacy)
        try fixture.write(gliner.files, to: aliased)
        try FileManager.default.createSymbolicLink(
            at: aliased.appending(path: "latest.aimodel"), withDestinationURL: aliased.appending(path: "x.aimodel"))
        try fixture.write(
            ["x.aimodel/main.mlirb.part": Data("partial".utf8), "contract.json": Data("{".utf8)],
            to: fixture.stagingArea(gliner))
        let originals = fixture.contents(documents)

        let first = await store.migrate([unknown, partial, linked, aliased, bundled, legacy], gliner)
        #expect(
            first.outcomes[unknown]?.undetermined
                == describe(ManagedModelStoreError.checksumMismatch("x.aimodel/main.mlirb")))
        #expect(
            first.outcomes[partial]?.undetermined
                == describe(ManagedModelStoreError.missingFile("tokenizer/tokenizer.json")))
        #expect(first.outcomes[linked]?.undetermined == describe(ManagedModelStoreError.symbolicLink))
        #expect(first.outcomes[aliased]?.undetermined == describe(ManagedModelStoreError.symbolicLink))
        let migrated = try #require(first.outcomes[bundled]?.resource)
        #expect(first.outcomes[legacy]?.isSkipped == true)
        #expect(try await store.inspect(gliner).resource == migrated)
        #expect(
            Set(fixture.contents(migrated.directory).filter { $0.value != "directory" }.keys)
                == Set(gliner.files.keys))
        for sidecar in ["._README.md", "x.aimodel/._main.mlirb"] {
            #expect(FileManager.default.fileExists(atPath: bundled.appending(path: sidecar).path))
            #expect(!FileManager.default.fileExists(atPath: migrated.directory.appending(path: sidecar).path))
        }

        let second = await store.migrate([unknown, legacy, bundled], gliner)
        #expect(second.outcomes.count == 3 && second.outcomes.values.allSatisfy(\.isSkipped))
        let revisions = fixture.root.appending(path: "artifacts/\(gliner.artifact.id)")
        #expect(try FileManager.default.contentsOfDirectory(atPath: revisions.path) == [distribution])
        #expect(!FileManager.default.fileExists(atPath: fixture.stagingArea(gliner).path))
        #expect(try fixture.ledger().entries.filter { $0.artifactID == gliner.artifact.id }.count == 1)
        #expect(await store.generation(of: migrated.identity) == 1)

        let embeddingReport = await store.migrate([legacyEmbedding], embedding)
        let migratedEmbedding = try #require(embeddingReport.outcomes[legacyEmbedding]?.resource)
        #expect(migratedEmbedding.matches(model: embedding.model, artifact: embedding.artifact))

        #expect(try await store.delete(identity: migrated.identity) == migrated.sizeBytes)
        #expect(await store.migrate([legacy], gliner).outcomes[legacy]?.resource == migrated)
        await #expect(throws: ManagedModelStoreError.pathEscape) { try await store.promote(legacy, gliner) }
        #expect(fixture.contents(documents) == originals)
    }

    @Test(
        "STORE-04 Deletion is refused while in use or for another identity and keeps inputs, saved results, legacy originals and other models"
    )
    func deletionBoundaries() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let gliner = try Sample.gliner()
        let embedding = try Sample.embedding()
        let installed = try await fixture.install(gliner, into: store)
        let other = try await fixture.install(embedding, into: store)
        try fixture.write(gliner.files, to: fixture.documents.appending(path: "gliner-fp32"))
        try fixture.write(["input.txt": Data("input".utf8)], to: fixture.documents.appending(path: "inputs"))
        try fixture.write(["result.json": Data("{}".utf8)], to: fixture.documents.appending(path: "results"))
        let caches = fixture.base.appending(path: "Caches")
        try fixture.write(["specialized.bin": Data("cache".utf8)], to: caches)
        let leftover = try await store.staging(gliner)
        try fixture.write(["partial.bin": Data("partial".utf8)], to: leftover)
        let kept = [fixture.contents(fixture.documents), fixture.contents(caches)]

        let lease = await store.lease(identity: installed.identity, owner: UUID())
        await #expect(throws: ManagedModelStoreError.inUse) { try await store.delete(identity: installed.identity) }
        #expect(await store.activeLeases(identity: installed.identity) == 1)
        for foreign in [
            identity(installed, revision: updatedDistribution),
            identity(installed, contract: "gliner-gl4-fp16-t256-t512"),
            identity(installed, modelID: embedding.model.id),
        ] {
            await #expect(throws: ManagedModelStoreError.notInstalled) { try await store.delete(identity: foreign) }
        }
        #expect(try await store.inspect(gliner).resource == installed)

        await lease.release()
        await lease.release()
        #expect(await store.activeLeases(identity: installed.identity) == 0)
        let bytes = installed.sizeBytes + Int64("partial".utf8.count)
        #expect(await store.deletionSize(identity: installed.identity) == bytes)
        #expect(try await store.delete(identity: installed.identity) == bytes)
        #expect(try await store.inspect(gliner).isMissing)
        #expect(!FileManager.default.fileExists(atPath: fixture.artifactDirectory(gliner).path))
        #expect(!FileManager.default.fileExists(atPath: fixture.stagingArea(gliner).path))
        #expect(try await store.inspect(embedding).resource == other)
        #expect([fixture.contents(fixture.documents), fixture.contents(caches)] == kept)
        #expect(try fixture.ledger().entries.map(\.artifactID) == [embedding.artifact.id])
        await #expect(throws: ManagedModelStoreError.notInstalled) {
            try await store.delete(identity: installed.identity)
        }
    }

    @Test("STORE-05 A deletion that fails midway is detected by re-inspection and never becomes ready")
    func failedDeletion() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let sample = try Sample.gliner()
        let installed = try await fixture.install(sample, into: store)
        let refusing = fixture.store(writeLedger: { _, _ in throw CocoaError(.fileWriteNoPermission) })
        await #expect(throws: CocoaError(.fileWriteNoPermission)) {
            try await refusing.delete(identity: installed.identity)
        }
        #expect(try await store.inspect(sample).resource == installed)

        try FileManager.default.setAttributes(
            [.posixPermissions: 0o555],
            ofItemAtPath: fixture.artifactDirectory(sample).appending(path: "x.aimodel").path)
        let failure = await #expect(throws: CocoaError.self) { try await store.delete(identity: installed.identity) }
        #expect(failure?.code == .fileWriteNoPermission)
        #expect(try await store.inspect(sample).isMissing)
        #expect(try await fixture.store().inspect(sample).isMissing)
        #expect(await store.generation(of: installed.identity) == nil)
    }

    @Test(
        "STORE-05 External replacement and stale or corrupted ledgers are detected by re-inspection and never permit execution"
    )
    func externalReplacement() async throws {
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let sample = try Sample.gliner()
        let directory = fixture.artifactDirectory(sample)
        let installed = try await fixture.install(sample, into: store)

        try Data("fixture-gliner-weightZ".utf8).write(to: directory.appending(path: "x.aimodel/main.mlirb"))
        #expect(
            try await store.inspect(sample).reason
                == describe(ManagedModelStoreError.checksumMismatch("x.aimodel/main.mlirb")))
        #expect(try fixture.ledger().entries.contains { $0.artifactID == sample.artifact.id })

        let copy = fixture.base.appending(path: "copy")
        try fixture.write(sample.files, to: copy)
        try FileManager.default.removeItem(at: directory)
        try FileManager.default.createSymbolicLink(at: directory, withDestinationURL: copy)
        #expect(try await store.inspect(sample).reason == describe(ManagedModelStoreError.symbolicLink))
        try FileManager.default.removeItem(at: directory)
        #expect(try await store.inspect(sample).reason == describe(ManagedModelStoreError.missingFile("contract.json")))

        #expect(try await fixture.install(sample, into: store) == installed)
        try Data("garbage".utf8).write(to: fixture.root.appending(path: "ledger.json"))
        await #expect(throws: ManagedModelStoreError.ledgerCorrupted) {
            try await store.delete(identity: installed.identity)
        }
        #expect(try await store.inspect(sample).resource == installed)
        #expect(try fixture.ledger().entries.map(\.artifactID) == [sample.artifact.id])

        let forged = try Sample.embedding()
        var ledger = try fixture.ledger()
        ledger.entries.append(
            Ledger.Entry(
                modelID: forged.model.id, artifactID: forged.artifact.id, contract: forged.artifact.contract,
                revision: distribution, upstreamRevision: embeddingSource.revision, sizeBytes: 1, installedAt: .now,
                generation: 99))
        try JSONEncoder().encode(ledger).write(to: fixture.root.appending(path: "ledger.json"))
        #expect(
            try await store.inspect(forged).reason == describe(ManagedModelStoreError.missingFile("MODEL_LICENSE.txt")))
        #expect(try await store.delete(identity: installed.identity) == installed.sizeBytes)
    }
}
