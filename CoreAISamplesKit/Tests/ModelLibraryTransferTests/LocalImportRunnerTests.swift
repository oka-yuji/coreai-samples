import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTestSupport
import ModelLibraryTransfer
import Testing

@Suite(.timeLimit(.minutes(1)))
@MainActor
struct LocalImportRunnerTests {
    @Test(
        "QMI-03 Cancellation or Store destruction at the copy boundary removes staging and preserves the installation",
        arguments: [false, true])
    func cancellationAtHandoff(destroyStore: Bool) async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let storage = fixture.store()
        let catalog = try IOSDemoCatalog.make(localImageArtifacts: [fixture.artifact])
        let model = try #require(catalog.model(containing: fixture.artifact.id))
        let stage = try await storage.copyForImport(
            from: fixture.source, model: model, artifact: fixture.artifact, progress: { _ in })
        let installed = try await storage.promote(staging: stage, model: model, artifact: fixture.artifact)
        let runner = ModelLibraryResourceRunner(store: storage, transfer: ArtifactTransfer())
        let (acknowledgements, sink) = AsyncStream.makeStream(of: Bool.self)
        defer { sink.finish() }
        let lifetime = ImportLibraryLifetime()
        defer { lifetime.library = nil }
        var validations = 0
        var copied = false
        lifetime.library = Store<ModelLibraryFeature>(
            initialState: ModelLibraryState(
                catalog: catalog, environment: LibraryEnvironment(platform: .iOS), owner: UUID())
        ) { [weak lifetime] effect, send in
            if case .validate = effect { validations += 1 }
            await runner.run(effect) { action in
                send(action)
                if case .importCopied = action {
                    copied = true
                    if destroyStore {
                        lifetime?.library = nil
                    } else {
                        lifetime?.library?.send(.cancelRequested(fixture.artifact.id))
                    }
                }
            }
            switch effect {
            case .importBundle where destroyStore, .stopImport:
                sink.yield(true)
                sink.finish()
            default: break
            }
        }
        lifetime.library?.send(.importRequested(fixture.artifact.id, fixture.source))
        var iterator = acknowledgements.makeAsyncIterator()
        #expect(await iterator.next() == true)
        #expect(copied && validations == 0)
        if !destroyStore {
            #expect(lifetime.library?.state.installations[fixture.artifact.id]?.phase == .ready)
        }
        guard case .ready(let restored) = try await storage.inspect(model: model, artifact: fixture.artifact) else {
            Issue.record("Stopping the import lost the installed model")
            return
        }
        #expect(restored == installed)
        let staging = try await storage.stagingDirectory(for: model, artifact: fixture.artifact)
        #expect(try FileManager.default.contentsOfDirectory(atPath: staging.path).isEmpty)
    }

    @Test("QMI-06 The import runner copies and validates a known bundle without executing a model")
    func importAndValidate() async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let runner = ModelLibraryResourceRunner(store: store, transfer: ArtifactTransfer())
        let ticket = LibraryTicket(owner: UUID(), number: 1)
        var copied = false
        var resource: ValidatedModelResource?
        var updates: [ModelTransferProgress] = []
        await runner.run(.importBundle(fixture.model, fixture.artifact, ticket, fixture.source)) { action in
            switch action {
            case .progress(let id, let request, let progress):
                #expect(id == fixture.artifact.id && request == ticket)
                updates.append(progress)
            case .importCopied(let id, let request):
                #expect(id == fixture.artifact.id && request == ticket)
                #expect(resource == nil)
                copied = true
            case .validated(let id, let request, let value):
                #expect(id == fixture.artifact.id && request == ticket && copied)
                resource = value
            default: Issue.record("Unexpected import action")
            }
        }
        #expect(updates.last?.completedBytes == fixture.artifact.totalFileBytes)
        #expect(updates.map(\.completedBytes) == updates.map(\.completedBytes).sorted())
        let validated = try #require(resource)
        #expect(validated.matches(model: fixture.model, artifact: fixture.artifact))
        let staging = try await store.stagingDirectory(for: fixture.model, artifact: fixture.artifact)
        #expect(try FileManager.default.contentsOfDirectory(atPath: staging.path).isEmpty)
    }

    @Test("QMI-03 A rejected import reports failure and its stop acknowledgement re-inspects the original installation")
    func rejectionAndStopInspection() async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let installed = try await fixture.install(into: store)
        let weights = fixture.source.appending(path: "text/text.aimodel/weights.bin")
        try Data(repeating: 9, count: 4096).write(to: weights)
        let runner = ModelLibraryResourceRunner(store: store, transfer: ArtifactTransfer())
        let ticket = LibraryTicket(owner: UUID(), number: 1)
        var failures: [ModelLibraryFailure] = []
        await runner.run(.importBundle(fixture.model, fixture.artifact, ticket, fixture.source)) { action in
            if case .failed(let id, let request, let error) = action {
                #expect(id == fixture.artifact.id && request == ticket)
                failures.append(error)
            } else if case .progress = action {
            } else {
                Issue.record("A rejected import produced a completion")
            }
        }
        #expect(failures == [.invalidResource])
        var restored: ValidatedModelResource?
        await runner.run(.stopImport(fixture.model, fixture.artifact, ticket)) { action in
            guard case .importStopped(let id, let request, let resource) = action else {
                Issue.record("Import stop did not inspect the installed model")
                return
            }
            #expect(id == fixture.artifact.id && request == ticket)
            restored = resource
        }
        #expect(restored == installed)
    }

    @Test("QMI-04 Failed promotion removes only its own staged import and preserves the installed resource")
    func failedPromotionCleansStaging() async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let installed = try await fixture.install(into: fixture.store())
        let store = ManagedModelStore(root: fixture.root, writeLedger: { _, _ in throw CocoaError(.fileWriteUnknown) })
        let runner = ModelLibraryResourceRunner(store: store, transfer: ArtifactTransfer())
        let ticket = LibraryTicket(owner: UUID(), number: 1)
        var failed = false
        var copied = false
        await runner.run(.importBundle(fixture.model, fixture.artifact, ticket, fixture.source)) { action in
            if case .failed(let id, let request, _) = action {
                #expect(id == fixture.artifact.id && request == ticket)
                failed = true
            } else if case .importCopied = action {
                copied = true
            } else if case .progress = action {
            } else {
                Issue.record("A failed promotion reported success")
            }
        }
        #expect(failed && copied)
        let staging = try await store.stagingDirectory(for: fixture.model, artifact: fixture.artifact)
        #expect(try FileManager.default.contentsOfDirectory(atPath: staging.path).isEmpty)
        guard case .ready(let current) = try await store.inspect(model: fixture.model, artifact: fixture.artifact)
        else {
            Issue.record("Failed promotion lost the existing resource")
            return
        }
        #expect(current == installed)
    }
}

@MainActor
private final class ImportLibraryLifetime {
    var library: Store<ModelLibraryFeature>?
}
