import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import Testing

@MainActor
struct MigrationCleanupTests {
    @Test("LIGHT-03 An inaccessible cleanup namespace preserves a fully verified destination")
    func inaccessibleJournal() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let managed = root.appending(path: "managed")
        let source = root.appending(path: "legacy")
        let sample = try Sample()
        try write(sample.files, to: source)
        let store = ManagedModelStore(root: managed, removeMigratedSources: true)
        let staging = try await store.stagingDirectory(for: sample.model, artifact: sample.artifact)
        try write(sample.files, to: staging)
        let installed = try await store.promote(staging: staging, model: sample.model, artifact: sample.artifact)
        try FileManager.default.createSymbolicLink(
            at: managed.appending(path: "migrations"), withDestinationURL: source)
        let runner = ModelLibraryResourceRunner(
            store: store, transfer: ArtifactTransfer(), migrationCandidates: { [source] })
        var actions: [ModelLibraryAction] = []
        await runner.run(.inspect(sample.model, sample.artifact, LibraryTicket(owner: UUID(), number: 1))) {
            actions.append($0)
        }
        guard case .inspected(_, _, let result) = try #require(actions.last) else {
            Issue.record("The verified model must remain available when its cleanup namespace is inaccessible.")
            return
        }
        #expect(result == installed)
        for (path, bytes) in sample.files {
            #expect(try Data(contentsOf: source.appending(path: path)) == bytes)
        }
    }

    @Test(
        "LIGHT-03 Resuming migration rejects a changed destination and preserves its source and journal",
        arguments: [false, true])
    func changedDestination(retired: Bool) async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let managed = root.appending(path: "managed")
        let source = root.appending(path: "legacy")
        let sample = try Sample()
        try write(sample.files, to: source)
        let store = ManagedModelStore(
            root: managed, removeMigratedSources: true,
            writeLedger: {
                try $0.write(to: $1, options: .atomic)
                withUnsafeCurrentTask { $0?.cancel() }
            })
        _ = await Task {
            await store.migrate(candidates: [source], model: sample.model, artifact: sample.artifact)
        }.value
        let area = managed.appending(path: "migrations/\(sample.id)/\(revision)")
        let journal = area.appending(path: "cleanup.json")
        #expect(FileManager.default.fileExists(atPath: journal.path))
        let retained = retired ? area.appending(path: "source") : source
        if retired { try FileManager.default.moveItem(at: source, to: retained) }
        let resource = try #require(try await store.inspect(model: sample.model, artifact: sample.artifact).resource)
        let file = resource.directory.appending(path: "embedding-manifest.json")
        var data = try Data(contentsOf: file)
        data[0] ^= 1
        try data.write(to: file)
        let runner = ModelLibraryResourceRunner(
            store: store, transfer: ArtifactTransfer(), migrationCandidates: { [source] })
        var actions: [ModelLibraryAction] = []
        await runner.run(.inspect(sample.model, sample.artifact, LibraryTicket(owner: UUID(), number: 1))) {
            actions.append($0)
        }
        guard case .inspected(_, _, let result) = try #require(actions.last) else {
            Issue.record("The changed destination must not become ready.")
            return
        }
        #expect(result == nil)
        #expect(FileManager.default.fileExists(atPath: journal.path))
        for (path, bytes) in sample.files {
            #expect(try Data(contentsOf: retained.appending(path: path)) == bytes)
        }
    }

    @Test("A cleanup failure preserves the verified model and the original folder for retry", arguments: [false, true])
    func readyDespiteCleanupFailure(resuming: Bool) async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let managed = root.appending(path: "managed")
        let source = root.appending(path: "legacy")
        let sample = try Sample()
        try write(sample.files, to: source)
        let store = ManagedModelStore(
            root: managed, removeMigratedSources: true,
            writeLedger: {
                try $0.write(to: $1, options: .atomic)
                try Data("keep".utf8).write(to: source.appending(path: "my-notes.txt"))
                if resuming { withUnsafeCurrentTask { $0?.cancel() } }
            })
        if resuming {
            _ = await Task {
                await store.migrate(candidates: [source], model: sample.model, artifact: sample.artifact)
            }.value
        }
        let runner = ModelLibraryResourceRunner(
            store: store, transfer: ArtifactTransfer(), migrationCandidates: { [source] })
        let ticket = LibraryTicket(owner: UUID(), number: 1)
        var actions: [ModelLibraryAction] = []
        await runner.run(.inspect(sample.model, sample.artifact, ticket)) { actions.append($0) }
        guard case .inspected(let id, let returned, let resource) = try #require(actions.last) else {
            Issue.record("The verified model must remain available while cleanup is pending.")
            return
        }
        #expect(id == sample.id && returned == ticket && resource != nil)
        #expect(try String(contentsOf: source.appending(path: "my-notes.txt"), encoding: .utf8) == "keep")
        #expect(
            FileManager.default.fileExists(
                atPath: managed.appending(path: "migrations/\(sample.id)/\(revision)/cleanup.json").path))
    }
}
