import ChronosRuntime
import ForecastFeature
import Foundation
import Testing

@testable import ForecastRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ForecastRunnerTests {
    @Test(
        "S2 Stop joins noncooperative loading and prediction before every release acknowledgement",
        arguments: [false, true])
    func stopJoins(_ predicting: Bool) async throws {
        let fake = try ForecastServiceFake()
        fake.pause = predicting ? "predict" : "load"
        let request = try fake.dataset.request(seriesID: "B1_1", originIndex: 365, horizon: 14)
        let runner = ForecastRunner(service: fake)
        var events = fake.events.stream.makeAsyncIterator()
        var delivered = 0
        let operation = Task {
            await runner.run(predicting ? .predict(1, request) : .load(1, refresh: false)) { _ in delivered += 1 }
        }
        #expect(await events.next() == fake.pause)
        let starts = AsyncStream<Void>.makeStream()
        var started = starts.stream.makeAsyncIterator()
        let first = Task {
            starts.continuation.yield(())
            await runner.release()
        }
        let second = Task {
            starts.continuation.yield(())
            await runner.release()
        }
        _ = await started.next()
        _ = await started.next()
        #expect(fake.releases == 0)
        operation.cancel()
        fake.finish()
        await operation.value
        await first.value
        await second.value
        #expect(delivered == 0 && fake.releases == 1 && !fake.releasedWhileBusy && fake.continuation == nil)
        starts.continuation.finish()
    }

    @Test("S2 Native errors remain errors and explicit retry can succeed without a fallback", arguments: [false, true])
    func failure(_ predicting: Bool) async throws {
        let fake = try ForecastServiceFake()
        let runner = ForecastRunner(service: fake)
        let request = try fake.dataset.request(seriesID: "B1_1", originIndex: 365, horizon: 14)
        fake.failing = true
        var failure: ForecastFailure?
        await runner.run(predicting ? .predict(4, request) : .load(4, refresh: false)) {
            if case .failed(4, let error) = $0 { failure = error }
        }
        #expect(failure == .predictionFailed)
        #expect(fake.loads + fake.predictions == 1)
        fake.failing = false
        var succeeded = false
        await runner.run(predicting ? .predict(5, request) : .load(5, refresh: true)) {
            switch $0 {
            case .predicted(5, _), .loaded(5, _): succeeded = true
            default: break
            }
        }
        #expect(succeeded)
        await runner.release()
    }

    @Test("S2 Revoked permissions reject both delayed success and delayed failure", arguments: [false, true])
    func revoked(_ failing: Bool) async throws {
        let fake = try ForecastServiceFake()
        fake.pause = "load"
        fake.failing = failing
        var permitted = true
        let runner = ForecastRunner(service: fake, permits: { permitted })
        var events = fake.events.stream.makeAsyncIterator()
        var count = 0
        let operation = Task { await runner.run(.load(1, refresh: false)) { _ in count += 1 } }
        #expect(await events.next() == "load")
        permitted = false
        fake.finish()
        await operation.value
        #expect(count == 0)
        await runner.release()
    }

    @Test(
        "S2 B2 Sessions revoke before joining and a delayed old close cannot release a newer service",
        arguments: [ForecastKind.productSales, .seoulBike])
    func sessions(_ kind: ForecastKind) async throws {
        let dataset =
            try await kind == .seoulBike ? BikeDataRepository().load() : SalesDataset.parse(ForecastFixture.csv())
        let first = try ForecastServiceFake(dataset: dataset)
        let second = try ForecastServiceFake(dataset: dataset)
        var created = 0
        var permission: (@MainActor () -> Bool)?
        let sessions = ForecastDemoSessions(
            kind: kind,
            makeService: { _, permits in
                created += 1
                if created == 1 {
                    permission = permits
                    return first
                }
                return second
            })
        let old = ForecastFixture.session(1, kind: kind)
        let new = ForecastFixture.session(2, kind: kind)
        let bound = sessions.bind(old, permits: { true })
        let store = try #require(bound)
        #expect(store.state.phase == .idle && first.loads == 0 && first.predictions == 0)
        first.pause = "load"
        var events = first.events.stream.makeAsyncIterator()
        store.send(.loadData)
        #expect(await events.next() == "load")
        let starts = AsyncStream<Void>.makeStream()
        var started = starts.stream.makeAsyncIterator()
        let stopping = Task {
            starts.continuation.yield(())
            await sessions.stop(old)
        }
        _ = await started.next()
        #expect(permission?() == false && store.state.ended)
        #expect(sessions.bind(old, permits: { true }) == nil && sessions.bind(new, permits: { true }) == nil)
        first.finish()
        await stopping.value
        #expect(first.releases >= 1 && !first.releasedWhileBusy)
        let rebound = sessions.bind(new, permits: { true })
        let next = try #require(rebound)
        await sessions.stop(old)
        store.send(.appear)
        #expect(next.state.phase == .idle && !next.state.ended && second.releases == 0)
        #expect(store.state.dataset == nil && created == 2)
        await sessions.stop(new)
        #expect(next.state.ended && second.releases >= 1)
        starts.continuation.finish()
    }

    @Test("S2 Live service authorizes the captured slice and preserves missing inputs and all raw outputs")
    func liveBoundary() async throws {
        let root = try ForecastFixture.directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let resource = ForecastFixture.session(1).resource
        let predictor = ForecastPredictorFake(identity: ForecastFixture.identity())
        var factories = 0
        let service = LiveForecastService(
            resource: resource, repository: ForecastFixture.repository(root),
            makeRuntime: { url, hash in
                #expect(url == resource.directory && hash == resource.revision)
                factories += 1
                return ChronosRuntime(factory: { predictor })
            })
        let dataset = try await service.load(refresh: false)
        #expect(factories == 0)
        let request = try dataset.request(seriesID: "B1_1", originIndex: 365, horizon: 14)
        let result = try await service.predict(request)
        let context = try #require(await predictor.contexts.first)
        #expect(context.count == 368 && context.prefix(3).allSatisfy(\.isNaN))
        #expect(
            zip(context.suffix(365), request.context).allSatisfy { raw, expected in
                expected.map { raw == $0 } ?? raw.isNaN
            })
        #expect(result.raw.quantiles.count == 21 && result.raw.quantiles.allSatisfy { $0.count == 16 && $0[0] == -2 })
        #expect(result.request == request && result.identity.manifest.sha256 == resource.revision)
        #expect(result.seconds >= 0 && result.loaded && factories == 1)
        await service.release()
        _ = try await service.predict(request)
        #expect(factories == 2)
        await service.release()
    }

    @Test("S2 Preview refuses native work and a mismatched runtime identity is rejected")
    func identity() async throws {
        let root = try ForecastFixture.directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = ForecastFixture.repository(root)
        let preview = LiveForecastService(repository: repository)
        let dataset = try await preview.load(refresh: false)
        let request = try dataset.request(seriesID: "B1_1", originIndex: 365, horizon: 14)
        await #expect(throws: ForecastFailure.unavailable) { try await preview.predict(request) }
        let predictor = ForecastPredictorFake(identity: ForecastFixture.identity(String(repeating: "b", count: 64)))
        let service = LiveForecastService(
            resource: ForecastFixture.session(1).resource, repository: repository,
            makeRuntime: { _, _ in ChronosRuntime(factory: { predictor }) })
        _ = try await service.load(refresh: false)
        await #expect(throws: ForecastFailure.predictionFailed) { try await service.predict(request) }
        await service.release()
        await preview.release()
    }
}
