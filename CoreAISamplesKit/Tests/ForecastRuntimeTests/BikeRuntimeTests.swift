import ChronosRuntime
import ForecastFeature
import Foundation
import ModelLibraryFeature
import Testing

@testable import ForecastRuntime

@Suite(.timeLimit(.minutes(1)))
@MainActor
struct BikeRuntimeTests {
    @Test("B2 Bike service rejects another kind before creating a runtime and preserves all native fake outputs")
    func serviceKind() async throws {
        let sales = try SalesDataset.parse(ForecastFixture.csv())
        var creations = 0
        let wrong = LiveForecastService(
            kind: .seoulBike, resource: ForecastFixture.session(1).resource,
            repository: FixedForecastRepository(dataset: sales),
            makeRuntime: { _, _ in
                creations += 1
                return ChronosRuntime(factory: { ForecastPredictorFake(identity: ForecastFixture.identity()) })
            })
        await #expect(throws: ForecastFailure.invalidData) { try await wrong.load(refresh: false) }
        let salesRequest = try sales.request(seriesID: "B1_1", originIndex: 365, horizon: 14)
        await #expect(throws: ForecastFailure.invalidRequest) { try await wrong.predict(salesRequest) }
        #expect(creations == 0)
        await wrong.release()

        let predictor = ForecastPredictorFake(identity: ForecastFixture.identity())
        let service = LiveForecastService(
            kind: .seoulBike, resource: ForecastFixture.session(2, kind: .seoulBike).resource,
            makeRuntime: { _, _ in
                creations += 1
                return ChronosRuntime(factory: { predictor })
            })
        let dataset = try await service.load(refresh: false)
        #expect(creations == 0)
        await #expect(throws: ForecastFailure.invalidRequest) { try await service.predict(salesRequest) }
        let request = try dataset.request(seriesID: BikeDataset.seriesID, originIndex: 1000, horizon: 4)
        let result = try await service.predict(request)
        let contexts = await predictor.contexts
        #expect(contexts.count == 1 && contexts[0].map(Optional.some) == request.context)
        #expect(result.raw.quantiles.count == 21 && result.raw.quantiles.allSatisfy { $0.count == 16 && $0[0] == -2 })
        #expect(result.median == [0, 0, 0, 0] && result.request == request && result.metrics.count == 4)
        await service.release()
        let preview = LiveForecastService(kind: .seoulBike)
        #expect(try await preview.load(refresh: false) == dataset)
        await #expect(throws: ForecastFailure.unavailable) { try await preview.predict(request) }
        await preview.release()
    }

    @Test("B2 A kind-bound session refuses foreign selections and reports an injected foreign dataset visibly")
    func sessionsKind() async throws {
        let fake = try ForecastServiceFake()
        let actions = AsyncStream<Void>.makeStream()
        defer { actions.continuation.finish() }
        let sessions = ForecastDemoSessions(
            kind: .seoulBike, makeService: { _, _ in fake }, observe: { _, _ in actions.continuation.yield(()) })
        #expect(sessions.bind(ForecastFixture.session(1), permits: { true }) == nil)
        let bike = ForecastFixture.session(2, kind: .seoulBike)
        let bound = sessions.bind(bike, permits: { true })
        let store = try #require(bound)
        var completed = actions.stream.makeAsyncIterator()
        store.send(.loadData)
        _ = await completed.next()
        #expect(store.state.kind == .seoulBike && store.state.phase == .idle && store.state.error == .invalidData)
        #expect(store.state.dataset == nil && fake.predictions == 0)
        let forged = ModelDemoSession(
            id: bike.id, selection: ChronosDemoCatalog.salesSelection, resource: bike.resource)
        #expect(sessions.bind(forged, permits: { true }) == nil)
        await sessions.stop(forged)
        #expect(!store.state.ended && fake.releases == 0)
        await sessions.stop(bike)
        #expect(store.state.ended && fake.releases >= 1)
    }
}

private struct FixedForecastRepository: ForecastDataLoading {
    let dataset: ForecastDataset
    func load(refresh: Bool) async throws -> ForecastDataset { dataset }
}
