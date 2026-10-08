import Observation
import Testing

@testable import FeatureStore

private nonisolated enum Probe: Feature {
    struct State: Equatable, Sendable {
        var log: [String] = []
    }

    enum Action: Sendable {
        case log(String)
        case start(String, id: String? = nil)
        case stop(id: String)
        case stopAll
        case stopAllThenStart(String)
    }

    typealias Effect = String
    typealias EffectID = String

    static func reduce(_ state: inout State, _ action: Action) -> [Command<String, String>] {
        switch action {
        case .log(let entry):
            state.log.append(entry)
            return []
        case .start(let label, let id):
            return [.run(label, id: id)]
        case .stop(let id):
            return [.cancel(id)]
        case .stopAll:
            return [.cancelAll]
        case .stopAllThenStart(let label):
            return [.cancelAll, .run(label, id: nil)]
        }
    }
}

@MainActor
private final class FakeWork {
    enum Event: Hashable, Sendable {
        case started(String)
        case cancelled(String)
        case finished(String)
    }

    let events: AsyncStream<Event>
    private let report: AsyncStream<Event>.Continuation
    private var pending: [String: CheckedContinuation<[Probe.Action], Never>] = [:]

    init() {
        (events, report) = AsyncStream.makeStream()
    }

    func run(_ label: String, send: (Probe.Action) -> Void) async {
        let results = await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                pending[label] = continuation
                report.yield(.started(label))
            }
        } onCancel: { [report] in
            report.yield(.cancelled(label))
        }
        results.forEach(send)
        report.yield(.finished(label))
    }

    func complete(_ label: String, sending results: [Probe.Action] = []) {
        pending.removeValue(forKey: label)?.resume(returning: results)
    }
}

@MainActor
private final class Box<Value> {
    var value: Value
    init(_ value: Value) { self.value = value }
}

@MainActor
@Suite("Store: guarantees G1-G5, G7 and G8", .timeLimit(.minutes(1)))
struct StoreTests {
    @Test("G1: An action sent during a state-change notification is processed after the current one finishes")
    func g1_sendFromAnObserverDuringAStateChangeWaitsItsTurn() {
        let store = Store<Probe>(initialState: .init()) { _, _ in }
        let seenByObserver = Box<[String]?>(nil)
        withObservationTracking {
            _ = store.state
        } onChange: {

            MainActor.assumeIsolated {
                seenByObserver.value = store.state.log
                store.send(.log("from observer"))
            }
        }

        store.send(.log("first"))

        #expect(seenByObserver.value == [])
        #expect(store.state.log == ["first", "from observer"])
    }

    @Test("G1: Actions sent from effects are also processed in the order sent")
    func g1_actionsFromEffectsAreProcessedInTheOrderSent() async {
        let work = FakeWork()
        let store = Store<Probe>(initialState: .init(), run: work.run)
        var events = work.events.makeAsyncIterator()

        store.send(.start("a"))
        #expect(await events.next() == .started("a"))
        store.send(.log("direct"))
        work.complete("a", sending: [.log("a1"), .log("a2")])
        #expect(await events.next() == .finished("a"))

        #expect(store.state.log == ["direct", "a1", "a2"])
    }

    @Test("G2: A run with an id in use cancels the running effect before it starts")
    func g2_runWithAnIDInUseCancelsTheRunningEffectFirst() async {
        let work = FakeWork()
        let store = Store<Probe>(initialState: .init(), run: work.run)
        var events = work.events.makeAsyncIterator()

        store.send(.start("first", id: "x"))
        #expect(await events.next() == .started("first"))
        store.send(.start("second", id: "x"))
        #expect(await events.next() == .cancelled("first"))
        #expect(await events.next() == .started("second"))

        work.complete("first")
        #expect(await events.next() == .finished("first"))
        work.complete("second", sending: [.log("second")])
        #expect(await events.next() == .finished("second"))
        #expect(store.state.log == ["second"])
    }

    @Test("G3: Results that a cancelled effect returns later are dropped")
    func g3_resultsOfACancelledEffectAreDropped() async {
        let work = FakeWork()
        let store = Store<Probe>(initialState: .init(), run: work.run)
        var events = work.events.makeAsyncIterator()

        store.send(.start("a", id: "x"))
        #expect(await events.next() == .started("a"))
        store.send(.stop(id: "x"))
        #expect(await events.next() == .cancelled("a"))
        work.complete("a", sending: [.log("late")])
        #expect(await events.next() == .finished("a"))

        #expect(store.state.log == [])
    }

    @Test("G3: Results that a replaced effect returns later are dropped")
    func g3_resultsOfAReplacedEffectAreDropped() async {
        let work = FakeWork()
        let store = Store<Probe>(initialState: .init(), run: work.run)
        var events = work.events.makeAsyncIterator()

        store.send(.start("old", id: "x"))
        #expect(await events.next() == .started("old"))
        store.send(.start("new", id: "x"))
        #expect(await events.next() == .cancelled("old"))
        #expect(await events.next() == .started("new"))
        work.complete("old", sending: [.log("old")])
        #expect(await events.next() == .finished("old"))
        work.complete("new", sending: [.log("new")])
        #expect(await events.next() == .finished("new"))

        #expect(store.state.log == ["new"])
    }

    @Test("G4: Releasing the store cancels every effect in flight")
    func g4_releasingTheStoreCancelsEveryEffectInFlight() async {
        let work = FakeWork()
        var store: Store<Probe>? = Store(initialState: .init(), run: work.run)
        weak let released = store
        var events = work.events.makeAsyncIterator()
        store?.send(.start("a"))
        #expect(await events.next() == .started("a"))
        store?.send(.start("b", id: "x"))
        #expect(await events.next() == .started("b"))

        store = nil

        #expect(released == nil)
        let cancelled = [await events.next(), await events.next()]
        #expect(Set(cancelled) == [.cancelled("a"), .cancelled("b")])

        work.complete("a", sending: [.log("after release")])
        work.complete("b")
        let finished = [await events.next(), await events.next()]
        #expect(Set(finished) == [.finished("a"), .finished("b")])
    }

    @Test("G5: A state equal to the old one after reduce is not assigned and notifies nothing")
    func g5_anUnchangedStateIsNotAssigned() async {
        let store = Store<Probe>(initialState: .init()) { _, _ in }

        await confirmation("observers notified", expectedCount: 0) { notified in
            withObservationTracking {
                _ = store.state
            } onChange: {
                notified()
            }
            store.send(.stop(id: "nothing running"))
        }
        await confirmation("observers notified", expectedCount: 1) { notified in
            withObservationTracking {
                _ = store.state
            } onChange: {
                notified()
            }
            store.send(.log("changed"))
        }
    }

    @Test("G7: An effect cancelled before it starts never runs")
    func g7_anEffectCancelledBeforeItStartsNeverRuns() async {
        let work = FakeWork()
        let store = Store<Probe>(initialState: .init(), run: work.run)
        var events = work.events.makeAsyncIterator()

        store.send(.start("a", id: "x"))
        store.send(.stop(id: "x"))
        store.send(.start("b"))

        let first = await events.next()
        #expect(first == .started("b"))
        work.complete("b")
        #expect(await events.next() == .finished("b"))
    }

    @Test("G7: An effect replaced under the same id before it starts never runs")
    func g7_anEffectReplacedBeforeItStartsNeverRuns() async {
        let work = FakeWork()
        let store = Store<Probe>(initialState: .init(), run: work.run)
        var events = work.events.makeAsyncIterator()

        store.send(.start("a", id: "x"))
        store.send(.start("b", id: "x"))

        let first = await events.next()
        #expect(first == .started("b"))
        work.complete("b", sending: [.log("b")])
        #expect(await events.next() == .finished("b"))
        #expect(store.state.log == ["b"])
    }

    @Test("G8: Results that arrive after cancelAll are dropped, and effects started later run normally")
    func g8_cancelAllDropsLateResultsAndLaterEffectsRunNormally() async {
        let work = FakeWork()
        let store = Store<Probe>(initialState: .init(), run: work.run)
        var events = work.events.makeAsyncIterator()
        store.send(.start("a"))
        #expect(await events.next() == .started("a"))
        store.send(.start("b", id: "x"))
        #expect(await events.next() == .started("b"))

        store.send(.stopAll)

        let cancelled = [await events.next(), await events.next()]
        #expect(Set(cancelled) == [.cancelled("a"), .cancelled("b")])
        work.complete("a", sending: [.log("late a")])
        #expect(await events.next() == .finished("a"))
        work.complete("b", sending: [.log("late b")])
        #expect(await events.next() == .finished("b"))
        #expect(store.state.log == [])

        store.send(.start("c", id: "x"))
        #expect(await events.next() == .started("c"))
        work.complete("c", sending: [.log("c")])
        #expect(await events.next() == .finished("c"))
        #expect(store.state.log == ["c"])
    }

    @Test("G8: Commands are handled in the order returned, so a run after cancelAll is not cancelled")
    func g8_aRunAfterCancelAllInTheSameReduceIsNotCancelled() async {
        let work = FakeWork()
        let store = Store<Probe>(initialState: .init(), run: work.run)
        var events = work.events.makeAsyncIterator()
        store.send(.start("a"))
        #expect(await events.next() == .started("a"))

        store.send(.stopAllThenStart("b"))

        #expect(await events.next() == .cancelled("a"))
        #expect(await events.next() == .started("b"))
        work.complete("b", sending: [.log("b")])
        #expect(await events.next() == .finished("b"))
        work.complete("a")
        #expect(await events.next() == .finished("a"))
        #expect(store.state.log == ["b"])
    }
}
