import ChronosCore
import ChronosFeature
import ChronosRuntime
import FeatureStore
import Foundation
import ModelLibraryFeature
import Observation
import Testing

@testable import ChronosDemoRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ChronosRunnerTests {
    @Test("D3 Every stop waiter joins a noncooperative load or prediction before releasing", arguments: [false, true])
    func stopJoins(_ predicting: Bool) async throws {
        let fake = try await ChronosServiceFake.make()
        fake.pause = predicting ? "predict" : "load"
        let runner = ChronosRunner(service: fake)
        let request = try #require(fake.demo.catalog.request(scenarioID: "friday-evening", step: 0, mode: .live))
        var iterator = fake.events.stream.makeAsyncIterator()
        var results: [ChronosFeature.Action] = []
        let operation = Task { await runner.run(predicting ? .predict(1, request) : .load(1)) { results.append($0) } }
        #expect(await iterator.next() == fake.pause)
        let stops = AsyncStream<Void>.makeStream()
        let stop1 = Task {
            stops.continuation.yield(())
            await runner.release()
        }
        let stop2 = Task {
            stops.continuation.yield(())
            await runner.release()
        }
        var started = stops.stream.makeAsyncIterator()
        _ = await started.next()
        _ = await started.next()
        #expect(fake.releaseCount == 0)

        operation.cancel()
        fake.finish()
        await operation.value
        await stop1.value
        await stop2.value
        #expect(results.isEmpty && fake.releaseCount == 1 && !fake.releasedWhileBusy)
        #expect(fake.continuation == nil)
    }

    @Test("D4 Runner failures stay failures and never invoke a cached retry", arguments: [false, true])
    func failure(_ predicting: Bool) async throws {
        let fake = try await ChronosServiceFake.make()
        fake.failing = true
        let runner = ChronosRunner(service: fake)
        let request = try #require(fake.demo.catalog.request(scenarioID: "friday-evening", step: 0, mode: .live))
        var failure: ChronosDemoFailure?
        await runner.run(predicting ? .predict(4, request) : .load(4)) {
            if case .failed(4, let value) = $0 { failure = value }
        }
        #expect(failure == .predictionFailed && fake.requests.count == (predicting ? 1 : 0))
        #expect(fake.requests.allSatisfy { $0.mode == .live })
        await runner.release()
        #expect(fake.releaseCount == 1)
    }

    @Test("D3 Revoked permissions suppress delayed success and delayed failure", arguments: [false, true])
    func revoked(_ fail: Bool) async throws {
        let fake = try await ChronosServiceFake.make()
        fake.pause = "load"
        fake.failing = fail
        var permitted = true
        let runner = ChronosRunner(service: fake, permits: { permitted })
        var iterator = fake.events.stream.makeAsyncIterator()
        var actions = 0
        let operation = Task { await runner.run(.load(1)) { _ in actions += 1 } }
        #expect(await iterator.next() == "load")
        permitted = false
        fake.finish()
        await operation.value
        #expect(actions == 0)
        await runner.release()
    }

    @Test("D3 Owner replacement revokes permission before waiting and old stops cannot affect a new session")
    func owners() async throws {
        let first = try await ChronosServiceFake.make()
        let second = try await ChronosServiceFake.make()
        let actions = AsyncStream<ChronosFeature.Action>.makeStream()
        var creations = 0
        var firstPermission: (@MainActor () -> Bool)?
        let sessions = ChronosDemoSessions(
            makeService: { _, permission in
                creations += 1
                if creations == 1 {
                    firstPermission = permission
                    return first
                }
                return second
            }, observe: { _, action in actions.continuation.yield(action) })
        let old = session(number: 1)
        let fresh = session(number: 2)
        let bound = sessions.bind(old, permits: { true })
        let store = try #require(bound)
        var observed = actions.stream.makeAsyncIterator()
        _ = await observed.next()
        #expect(store.state.phase == .ready && first.requests.isEmpty)
        first.pause = "predict"
        var events = first.events.stream.makeAsyncIterator()
        #expect(await events.next() == "load")
        store.send(.predict)
        #expect(await events.next() == "predict")
        let stopping = AsyncStream<Void>.makeStream()
        let stop = Task {
            stopping.continuation.yield(())
            await sessions.stop(old)
        }
        var started = stopping.stream.makeAsyncIterator()
        _ = await started.next()
        #expect(firstPermission?() == false && store.state.ended)
        #expect(sessions.bind(old, permits: { true }) == nil)
        #expect(sessions.bind(fresh, permits: { true }) == nil)
        first.finish()
        await stop.value
        #expect(first.releaseCount >= 1 && !first.releasedWhileBusy)
        let rebound = sessions.bind(fresh, permits: { true })
        let next = try #require(rebound)
        while await observed.next() != nil { if next.state.phase == .ready { break } }
        await sessions.stop(old)
        store.send(.appear)
        #expect(next.state.phase == .ready && second.releaseCount == 0 && creations == 2)
        #expect(store.state.ended && store.state.forecast == nil)
        await sessions.stop(fresh)
        #expect(next.state.ended && second.releaseCount >= 1)
        actions.continuation.finish()
        stopping.continuation.finish()
    }

    @Test("D4 Cached preview has a Store lifetime and does not load native resources")
    func preview() async throws {
        let fake = try await ChronosServiceFake.make(live: false)
        let store = ChronosDemoSessions.preview(service: fake)
        store.send(.appear)
        for await phase in Observations({ store.state.phase }) { if phase == .ready { break } }
        #expect(store.state.mode == .precomputed && fake.requests.isEmpty)
        store.send(.predict)
        for await phase in Observations({ store.state.phase }) { if phase == .ready { break } }
        #expect(store.state.forecast?.mode == .precomputed && store.state.forecast?.processingSeconds == nil)
        store.send(.suspend)
        for await phase in Observations({ store.state.phase }) { if phase == .ready { break } }
        #expect(fake.releaseCount == 1 && !store.state.active)
        store.send(.appear)
        #expect(store.state.active && store.state.canPredict)
        var events = fake.events.stream.makeAsyncIterator()
        store.send(.ended)
        var releases = 0
        while let event = await events.next() {
            if event == "released" { releases += 1 }
            if releases == 2 { break }
        }
        #expect(fake.releaseCount == 2 && fake.continuation == nil)

    }

    @Test("D4 Live service retains every raw quantile and binds the managed identity before lazy loading")
    func fullRaw() async throws {
        let capture = ChronosCapture()
        let resource = session(number: 1).resource
        var factories = 0
        let service = LiveChronosService(
            resource: resource,
            makeRuntime: { url, manifest in
                #expect(url == resource.directory && manifest == resource.revision)
                factories += 1
                return ChronosRuntime(factory: { ChronosDemoPredictor(capture: capture, manifest: manifest) })
            })
        let loaded = try await service.load()
        #expect(factories == 0 && loaded.liveAvailable)
        let request = try #require(loaded.catalog.request(scenarioID: "friday-evening", step: 1, mode: .live))
        let batch = try await service.predict(request)
        let values = await capture.values
        #expect(values == request.series.map(\.values) && values.count == 20)
        #expect(batch.identity?.manifest.sha256 == resource.revision && batch.processingSeconds != nil)
        #expect(batch.areas.allSatisfy { $0.raw?.quantiles.count == 21 && $0.raw?.quantiles[20][15] == -1 })
        #expect(batch.areas.allSatisfy { $0.lower == [0, 0, 0, 0] })
        await service.release()
    }

    @Test("D3 Permission lost after one native series prevents the remaining nineteen")
    func seriesBoundary() async throws {
        let capture = ChronosCapture()
        var permissions = 0
        let service = LiveChronosService(
            resource: session(number: 1).resource,
            permits: {
                permissions += 1
                return permissions < 7
            },
            makeRuntime: { _, manifest in
                ChronosRuntime(factory: { ChronosDemoPredictor(capture: capture, manifest: manifest) })
            })
        let loaded = try await service.load()
        let request = try #require(loaded.catalog.request(scenarioID: "friday-evening", step: 0, mode: .live))
        await #expect(throws: CancellationError.self) { try await service.predict(request) }
        #expect(await capture.values.count == 1)
        await service.release()
    }

    private func session(number: Int) -> ModelDemoSession {
        ModelDemoSession(
            id: LibraryTicket(owner: UUID(), number: number), selection: ChronosDemoCatalog.selection,
            resource: ValidatedModelResource(
                modelID: "chronos2", artifactID: "chronos2-fp32", contract: "chronos2-fp32-v1",
                revision: "managed-manifest", upstreamRevision: ChronosContract.revision,
                directory: URL(filePath: "/unused-chronos"), sizeBytes: 1, sourceKind: .localManifest))
    }
}

@MainActor
private final class ChronosServiceFake: ChronosService {
    let demo: ChronosLoadedDemo
    let repository: ChronosRepository
    let events = AsyncStream<String>.makeStream()
    var pause: String?
    var failing = false
    var continuation: CheckedContinuation<Void, Never>?
    var requests: [ChronosForecastRequest] = []
    var releaseCount = 0
    var busy = false
    var releasedWhileBusy = false

    init(demo: ChronosLoadedDemo, repository: ChronosRepository) {
        self.demo = demo
        self.repository = repository
    }
    static func make(live: Bool = true) async throws -> ChronosServiceFake {
        let repository = try ChronosRepository.bundled()
        return try await ChronosServiceFake(
            demo: ChronosLoadedDemo(catalog: repository.load(), liveAvailable: live), repository: repository)
    }
    func load() async throws -> ChronosLoadedDemo {
        busy = true
        defer { busy = false }
        await wait("load")
        if failing { throw ChronosDemoFailure.predictionFailed }
        return demo
    }
    func predict(_ request: ChronosForecastRequest) async throws -> ChronosForecastBatch {
        busy = true
        defer { busy = false }
        requests.append(request)
        await wait("predict")
        if failing { throw ChronosDemoFailure.predictionFailed }
        let cached = ChronosForecastRequest(
            scenarioID: request.scenarioID, originStep: request.originStep, mode: .precomputed, series: request.series)
        return try await repository.cached(cached)
    }
    func release() async {
        releasedWhileBusy = releasedWhileBusy || busy
        releaseCount += 1
        events.continuation.yield("released")
    }
    func finish() {
        let pending = continuation
        continuation = nil
        pending?.resume()
    }
    private func wait(_ event: String) async {
        if pause == event {
            await withCheckedContinuation {
                continuation = $0
                events.continuation.yield(event)
            }
        } else {
            events.continuation.yield(event)
        }
    }
}

private actor ChronosCapture {
    var values: [[Float]] = []
    func record(_ value: [Float]) { values.append(value) }
}

private struct ChronosDemoPredictor: ChronosPredictor {
    let capture: ChronosCapture
    let manifest: String
    var identity: ChronosBundleIdentity {
        ChronosBundleIdentity(
            source: .expected, manifest: .init(bytes: 1, sha256: manifest),
            files: ["Model.aimodel/main.mlirb": .init(bytes: 1, sha256: ChronosRepository.graphSHA256)])
    }
    func predict(_ context: ChronosContext) async throws -> ChronosPrediction {
        await capture.record(context.paddedValues)
        return try ChronosPrediction(quantiles: Array(repeating: Array(repeating: -1, count: 16), count: 21))
    }
}
