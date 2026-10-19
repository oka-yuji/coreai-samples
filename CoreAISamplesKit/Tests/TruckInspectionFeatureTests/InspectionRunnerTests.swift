import Foundation
import Testing
import TruckInspectionCore
import TruckInspectionFeature
import TruckInspectionRuntime

@MainActor
final class InspectionModelFake: TruckInspectionModelService {
    var failure = false
    var suspended = false
    var unloadFailure: InspectionError?
    var pending: CheckedContinuation<Void, Never>?
    var drains: [CheckedContinuation<Void, Never>] = []
    var unloads = 0
    var preparedFolders: [URL] = []
    var classifiedTexts: [String] = []
    let started = AsyncStream<Void>.makeStream()
    let draining = AsyncStream<Void>.makeStream()

    func prepare(folder: URL) async throws -> InspectionModelInfo {
        preparedFolders.append(folder)
        try await hold()
        if failure { throw InspectionError.invalidCatalog }
        return InspectionModelInfo(folder: folder, revision: "test", milliseconds: 5, computePreference: "gpu")
    }

    func classify(text: String, item: InspectionItem, catalog: InspectionCatalog) async throws -> InspectionCandidate {
        classifiedTexts.append(text)
        try await hold()
        if failure { throw InspectionError.invalidClassification }
        return InspectionCandidate(status: .attention, method: .model, score: 0.9)
    }

    func waitForIdle() async {
        draining.continuation.yield(())
        if pending != nil { await withCheckedContinuation { drains.append($0) } }
    }

    func unload() throws {
        if let unloadFailure { throw unloadFailure }
        guard pending == nil else { throw InspectionError.busy }
        unloads += 1
    }

    func finish() {
        let continuation = pending
        pending = nil
        continuation?.resume()
    }

    private func hold() async throws {
        started.continuation.yield(())
        if suspended { await withCheckedContinuation { pending = $0 } }
        let waiting = drains
        drains.removeAll()
        for continuation in waiting { continuation.resume() }
    }
}

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct InspectionRunnerTests {
    @Test("TT3: Model preparation and classification report success and failure")
    func results() async throws {
        let catalog = try InspectionCatalogLoader.load()
        let item = try #require(catalog.items.first)
        let model = InspectionModelFake()
        let runner = TruckInspectionRunner(model: model)
        var actions: [InspectionAction] = []
        await runner.run(.prepare(1, URL(filePath: "/asset"))) { actions.append($0) }
        guard case .modelLoaded(1, _) = actions.last else {
            Issue.record("Expected model result")
            return
        }
        await runner.run(.classify(2, "answer", item, catalog)) { actions.append($0) }
        guard case .classified(2, let candidate, _) = actions.last else {
            Issue.record("Expected candidate")
            return
        }
        #expect(candidate.needsConfirmation)
        model.failure = true
        await runner.run(.prepare(3, URL(filePath: "/bad"))) { actions.append($0) }
        guard case .failed(3, _) = actions.last else {
            Issue.record("Expected preparation failure")
            return
        }
        await runner.run(.classify(4, "answer", item, catalog)) { actions.append($0) }
        guard case .failed(4, _) = actions.last else {
            Issue.record("Expected classification failure")
            return
        }
        await runner.run(.stop(5, unload: true)) { actions.append($0) }
        #expect(model.unloads == 1)
        guard case .stopped(5, true) = actions.last else {
            Issue.record("Expected cleanup completion")
            return
        }
    }

    @Test("TT2: Cancellation suppresses late success and cleanup waits for the resource")
    func cancellationAndDrain() async {
        let model = InspectionModelFake()
        model.suspended = true
        let runner = TruckInspectionRunner(model: model)
        var actions: [InspectionAction] = []
        let operation = Task { await runner.run(.prepare(1, URL(filePath: "/asset"))) { actions.append($0) } }
        var starts = model.started.stream.makeAsyncIterator()
        _ = await starts.next()
        operation.cancel()
        let drain = Task { await runner.run(.stop(2, unload: true)) { actions.append($0) } }
        var drains = model.draining.stream.makeAsyncIterator()
        _ = await drains.next()
        #expect(model.unloads == 0)
        #expect(
            actions.allSatisfy { action in
                switch action {
                case .diagnostic, .voice(.clock): true
                default: false
                }
            })
        model.finish()
        await operation.value
        await drain.value
        let outcomes = actions.filter { action in
            switch action {
            case .diagnostic, .voice(.clock): false
            default: true
            }
        }
        #expect(outcomes.count == 1)
        guard case .stopped(2, true) = outcomes.last else {
            Issue.record("Expected cleanup completion")
            return
        }
        #expect(model.unloads == 1)
    }

    @Test("INSPECT-03: Release waits for a running classification before it unloads the idle model")
    func releaseJoinsClassification() async throws {
        let catalog = try InspectionCatalogLoader.load()
        let item = try #require(catalog.items.first)
        let model = InspectionModelFake()
        model.suspended = true
        let runner = TruckInspectionRunner(model: model)
        var actions: [InspectionAction] = []
        let operation = Task { await runner.run(.classify(1, "answer", item, catalog)) { actions.append($0) } }
        var starts = model.started.stream.makeAsyncIterator()
        _ = await starts.next()
        operation.cancel()
        var released = false
        let release = Task {
            try await runner.release()
            released = true
        }
        var drains = model.draining.stream.makeAsyncIterator()
        _ = await drains.next()
        #expect(!released)
        #expect(model.unloads == 0)
        model.finish()
        try await release.value
        await operation.value
        #expect(released)
        #expect(model.unloads == 1)
        #expect(!actions.contains { if case .classified = $0 { true } else { false } })
    }

    @Test("INSPECT-03: A model that cannot be unloaded reports a failed stop instead of an unloaded stop")
    func busyUnload() async {
        let model = InspectionModelFake()
        model.unloadFailure = .busy
        let runner = TruckInspectionRunner(model: model)
        var actions: [InspectionAction] = []
        await runner.run(.stop(1, unload: true)) { actions.append($0) }
        guard case .stopFailed(1, _) = actions.last else {
            Issue.record("Expected a failed stop")
            return
        }
        #expect(!actions.contains { if case .stopped = $0 { true } else { false } })
        await #expect(throws: InspectionReleaseError.self) { try await runner.release() }
        model.unloadFailure = nil
        await runner.run(.stop(2, unload: true)) { actions.append($0) }
        guard case .stopped(2, true) = actions.last else {
            Issue.record("Expected an unloaded stop after the retry")
            return
        }
        #expect(model.unloads == 1)
    }

    @Test("A missing model fails without starting model inference and releases its operation")
    func missingModel() async throws {
        let model = LiveTruckInspectionModel()
        let catalog = try InspectionCatalogLoader.load()
        let item = try #require(catalog.items.first)
        for _ in 0..<2 {
            await #expect(throws: InspectionError.noModel) {
                try await model.classify(text: "answer", item: item, catalog: catalog)
            }
            await model.waitForIdle()
        }
    }
}
