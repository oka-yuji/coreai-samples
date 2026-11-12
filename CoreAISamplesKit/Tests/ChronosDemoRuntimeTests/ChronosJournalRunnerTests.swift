import ChronosFeature
import FeatureStore
import Foundation
import Synchronization
import Testing

@testable import ChronosDemoRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ChronosJournalRunnerTests {
    @Test(
        "J2 Close back selection background and session end prevent late noncooperative actuals from being saved",
        arguments: ["close", "back", "select", "background", "ended"])
    func lateActuals(action: String) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ChronosAPIFixture.syntheticProfile(areas: NYCAPIContract.zoneIDs)
        let files = ChronosJournalFiles(directory: root.appending(path: "journal"), verification: profile)
        let record = try await ChronosJournalFixture.record(step: 1)
        try await files.create(record, permit: ChronosPublicationPermit())
        let window = try await ChronosJournalFixture.actuals(origin: record.input.snapshot.origin, profile: profile)
        let gate = ChronosJournalGate(window: window)
        let runner = ChronosJournalRunner(files: files, actuals: { try await gate.fetch($0, areas: $1) })
        let completed = AsyncStream<ChronosJournalFeature.Effect>.makeStream()
        defer {
            completed.continuation.finish()
            gate.started.continuation.finish()
        }
        let store = Store<ChronosJournalFeature>(initialState: ChronosJournalFeature.State()) { effect, send in
            await runner.run(effect, send: send)
            completed.continuation.yield(effect)
        }
        var effects = completed.stream.makeAsyncIterator()
        var starts = gate.started.stream.makeAsyncIterator()
        store.send(.open)
        store.send(.select(record.id))
        while await effects.next() != nil { if store.state.detail != nil { break } }
        store.send(.check)
        _ = await starts.next()
        switch action {
        case "close": store.send(.close)
        case "back": store.send(.back)
        case "select": store.send(.select(UUID()))
        case "background": store.send(.suspend)
        default: store.send(.ended)
        }
        await gate.finish()
        while let effect = await effects.next() { if case .check = effect { break } }
        #expect(try await files.detail(record.id).checks.isEmpty)
        #expect(await gate.requestedOrigin == record.input.snapshot.origin)
        #expect(await gate.requestedAreas == record.areas.map(\.areaID))
        #expect(await gate.continuation == nil)
        store.send(.ended)
    }

    @Test("J2 Save failures preserve native results and retries reuse the original UUID")
    func saveRetry() async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let writes = Mutex(0)
        let files = ChronosJournalFiles(
            directory: root.appending(path: "journal"),
            write: { data, file in
                let attempt = writes.withLock {
                    $0 += 1
                    return $0
                }
                if attempt == 1 { throw POSIXError(.ENOSPC) }
                try file.write(contentsOf: data)
            })
        let id = UUID()
        let runner = ChronosJournalRunner(
            files: files, actuals: { _, _ in throw ChronosDemoFailure.network },
            now: { ChronosAPIFixture.now }, makeID: { id }, device: { ChronosJournalFixture.device })
        let completed = AsyncStream<ChronosJournalFeature.Action>.makeStream()
        defer { completed.continuation.finish() }
        let journal = Store<ChronosJournalFeature>(initialState: ChronosJournalFeature.State()) { effect, send in
            await runner.run(effect) { action in
                send(action)
                completed.continuation.yield(action)
            }
        }
        let batch = try await ChronosJournalFixture.batch()
        let catalog = try await ChronosRepository.bundled().load()
        var feature = ChronosFeature.State()
        _ = ChronosFeature.reduce(&feature, .appear)
        _ = ChronosFeature.reduce(
            &feature, .loaded(feature.ticket, ChronosLoadedDemo(catalog: catalog, liveAvailable: true)))
        _ = ChronosFeature.reduce(&feature, .predict)
        let predictor = ChronosRunner(
            service: ChronosJournalPredictionStub(catalog: catalog, batch: batch), journal: journal)
        let request = try #require(catalog.request(scenarioID: batch.scenarioID, step: 0, mode: .live))
        await predictor.run(.predict(feature.ticket, request)) { action in
            _ = ChronosFeature.reduce(&feature, action)
        }
        let stable = feature
        var actions = completed.stream.makeAsyncIterator()
        _ = await actions.next()
        #expect(journal.state.saveError == .persistence && !journal.state.saving)
        #expect(try await files.list().isEmpty)
        journal.send(.retrySave)
        _ = await actions.next()
        #expect(journal.state.savedID == id && journal.state.saveError == nil)
        #expect(try await files.record(id).input == batch.capturedInput)
        journal.send(.retrySave)
        #expect(writes.withLock { $0 } == 2)
        #expect(feature == stable && feature.forecast == batch && feature.error == nil && feature.mode == .live)
        journal.send(.ended)
    }

    @Test(
        "J3 Network and rate limit failures remain errors without producing publication-pending checks",
        arguments: [ChronosDemoFailure.network, .rateLimited, .invalidData])
    func checkErrors(error: ChronosDemoFailure) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let files = ChronosJournalFiles(directory: root.appending(path: "journal"))
        let record = try await ChronosJournalFixture.record()
        try await files.create(record, permit: ChronosPublicationPermit())
        let runner = ChronosJournalRunner(files: files, actuals: { _, _ in throw error })
        var received: ChronosDemoFailure?
        await runner.run(.check(7, record.id)) { action in
            if case .failed(7, let failure) = action { received = failure }
        }
        #expect(received == error)
        #expect(try await files.detail(record.id).checks.isEmpty)
    }
}

@MainActor
private final class ChronosJournalPredictionStub: ChronosService {
    let catalog: ChronosCatalog
    let batch: ChronosForecastBatch

    init(catalog: ChronosCatalog, batch: ChronosForecastBatch) {
        self.catalog = catalog
        self.batch = batch
    }

    func load() async throws -> ChronosLoadedDemo { ChronosLoadedDemo(catalog: catalog, liveAvailable: true) }
    func predict(_ request: ChronosForecastRequest) async throws -> ChronosForecastBatch { batch }
    func release() async {}
}

private actor ChronosJournalGate {
    let window: APIWindow
    let started = AsyncStream<Void>.makeStream()
    var requestedOrigin: Date?
    var requestedAreas: [String]?
    var continuation: CheckedContinuation<Void, Never>?

    init(window: APIWindow) { self.window = window }

    func fetch(_ origin: Date, areas: [String]) async throws -> APIWindow {
        requestedOrigin = origin
        requestedAreas = areas
        started.continuation.yield(())
        await withCheckedContinuation { continuation = $0 }
        return window
    }

    func finish() {
        continuation?.resume()
        continuation = nil
    }
}
