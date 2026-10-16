import FeatureStore
import Foundation
import ObjectFinderCore
import ObjectFinderFeature
import Testing

@testable import ObjectFinderRuntime

@MainActor
struct ObjectFinderReplacementStoreTests {
    private struct Fixture {
        let store: Store<ObjectFinderFeature>
        let encoder: ReusedEncoder
        let events: AsyncStream<ObjectFinderAction>
        let continuation: AsyncStream<ObjectFinderAction>.Continuation
    }

    private func fixture(releaseFailure: Bool = false) throws -> Fixture {
        let encoder = try ReusedEncoder()
        let finder = ObjectRegionFinder(
            embedding: .init(
                load: { await encoder.load($0) }, text: { await encoder.text($0) },
                image: { _ in encoder.keys }, close: { await encoder.close() }), propose: { _ in [] })
        let camera = FinderDevicesFake()
        var devices = camera.devices
        devices.prepare = { try await finder.prepare(folder: $0, query: $1) }
        devices.closeFinder = {
            try await finder.close()
            if releaseFailure { throw ObjectFinderFailure.operation("release failed") }
        }
        let service = LiveObjectFinderService(devices: devices)
        let runner = ObjectFinderRunner(service: service)
        let events = AsyncStream<ObjectFinderAction>.makeStream(bufferingPolicy: .bufferingNewest(8))
        var state = ObjectFinderState(folder: URL(filePath: "/model"))
        state.query = "glasses"
        let store = Store<ObjectFinderFeature>(initialState: state) { effect, send in
            await runner.run(effect) { action in
                send(action)
                events.continuation.yield(action)
            }
        }
        return Fixture(store: store, encoder: encoder, events: events.stream, continuation: events.continuation)
    }

    @Test(
        "The real Store, Runner, Service and Finder reuse one load for two targets and release on Stop",
        .timeLimit(.minutes(1)))
    func replacementIntegration() async throws {
        let fixture = try fixture()
        var events = fixture.events.makeAsyncIterator()
        fixture.store.send(.start)
        _ = await events.next()
        #expect(fixture.store.state.phase == .searching)
        fixture.store.send(.queryChanged("keys"))
        fixture.store.send(.applyQuery)
        _ = await events.next()
        _ = await events.next()
        #expect(fixture.store.state.phase == .searching && fixture.store.state.session == 2)
        #expect(fixture.store.state.submittedQuery == "keys")
        #expect(await fixture.encoder.loads.count == 1)
        #expect(await fixture.encoder.queries == ["glasses", "keys"])
        #expect(await fixture.encoder.closeCancellation.isEmpty)
        fixture.store.send(.stop)
        _ = await events.next()
        #expect(fixture.store.state.phase == .idle)
        #expect(await fixture.encoder.closeCancellation == [false])
        fixture.continuation.finish()
    }

    @Test(
        "Stop, background and dismissal cancel queued replacement effects and release the real Finder",
        .timeLimit(.minutes(1)), arguments: [ObjectFinderAction.stop, .backgrounded, .disappeared])
    func cancellationIntegration(_ action: ObjectFinderAction) async throws {
        let fixture = try fixture()
        var events = fixture.events.makeAsyncIterator()
        fixture.store.send(.start)
        _ = await events.next()
        fixture.store.send(.queryChanged("keys"))
        fixture.store.send(.applyQuery)
        fixture.store.send(action)
        _ = await events.next()
        #expect(fixture.store.state.phase == .idle && !fixture.store.state.restartAfterStop)
        #expect(await fixture.encoder.loads.count == 1)
        #expect(await fixture.encoder.queries == ["glasses"])
        #expect(await fixture.encoder.closeCancellation == [false])
        fixture.continuation.finish()
    }

    @Test(
        "The runner reports cleanup failure without sending a successful stop acknowledgment", .timeLimit(.minutes(1)))
    func releaseFailureIntegration() async throws {
        let fixture = try fixture(releaseFailure: true)
        var events = fixture.events.makeAsyncIterator()
        fixture.store.send(.start)
        _ = await events.next()
        fixture.store.send(.stop)
        let action = await events.next()
        if case .stopFailed(1, .operation("release failed"))? = action {
        } else {
            Issue.record("Expected the control failure, not a successful stop")
        }
        #expect(fixture.store.state.phase == .idle && fixture.store.state.failure == .operation("release failed"))
        fixture.continuation.finish()
    }
}
