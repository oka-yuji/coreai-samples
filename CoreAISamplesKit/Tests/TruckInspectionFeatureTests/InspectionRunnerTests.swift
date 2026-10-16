import Foundation
import Testing
import TruckInspectionCore
import TruckInspectionFeature
import TruckInspectionRuntime

@MainActor
final class InspectionModelFake: TruckInspectionModelService {
    var failure = false
    var suspended = false
    var pending: CheckedContinuation<Void, Never>?
    var drains: [CheckedContinuation<Void, Never>] = []
    var unloads = 0
    var classifiedTexts: [String] = []
    let started = AsyncStream<Void>.makeStream()
    let draining = AsyncStream<Void>.makeStream()

    func prepare(folder: URL) async throws -> InspectionModelInfo {
        started.continuation.yield(())
        if suspended { await withCheckedContinuation { pending = $0 } }
        defer {
            let waiting = drains
            drains.removeAll()
            for continuation in waiting { continuation.resume() }
        }
        if failure { throw InspectionError.invalidCatalog }
        return InspectionModelInfo(folder: folder, revision: "test", milliseconds: 5, computePreference: "gpu")
    }

    func classify(text: String, item: InspectionItem, catalog: InspectionCatalog) async throws -> InspectionCandidate {
        classifiedTexts.append(text)
        if failure { throw InspectionError.invalidClassification }
        return InspectionCandidate(status: .attention, method: .model, score: 0.9)
    }

    func waitForIdle() async {
        draining.continuation.yield(())
        if pending != nil { await withCheckedContinuation { drains.append($0) } }
    }

    func unload() { unloads += 1 }

    func finish() {
        let continuation = pending
        pending = nil
        continuation?.resume()
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
