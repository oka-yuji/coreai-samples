import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import Testing

@MainActor
struct MigrationCleanupTests {
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
