import DecideFeature
import DecideRuntime
import FeatureStore
import Foundation
import Testing

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct DecideRunnerTests {
    @Test("Every service effect returns its successful action")
    func successes() async {
        let service = FakeDecideService()
        let runner = DecideRunner(service: service)
        var actions: [DecideAction] = []
        await runner.run(.load(1, "/bundle")) { actions.append($0) }
        #expect(actions == [.loaded(1, service.info)])
        actions = []
        await runner.run(.count(2, 3, service.input)) { actions.append($0) }
        #expect(actions == [.counted(2, 3, 42)])
        actions = []
        await runner.run(.read(4, URL(filePath: "/document.txt"))) { actions.append($0) }
        #expect(actions == [.documentRead(4, "Imported document")])
        actions = []
        await runner.run(.decide(5, 6, service.input, reuse: true)) { actions.append($0) }
        #expect(actions == [.completed(5, 6, service.output)])
        actions = []
        await runner.run(.examples(7)) { actions.append($0) }
        #expect(actions == [.examplesLoaded(7, [service.example])])
        actions = []
        await runner.run(.waitForIdle(8)) { actions.append($0) }
        #expect(actions == [.stopped(8)])
    }

    @Test("Service failures are reported as actions and cancellation is not shown as a failure")
    func failures() async {
        let service = FakeDecideService()
        let runner = DecideRunner(service: service)
        let effects: [(DecideEffect, DecideAction)] = [
            (.load(1, "/bundle"), .failed(1, "Expected failure.")),
            (.count(1, 2, service.input), .countFailed(1, 2, "Expected failure.")),
            (.read(1, URL(filePath: "/document.txt")), .failed(1, "Expected failure.")),
            (.decide(1, 2, service.input, reuse: true), .failed(1, "Expected failure.")),
            (.examples(1), .examplesFailed(1, "Expected failure.")),
        ]
        for (effect, expected) in effects {
            service.failure = .error
            var actions: [DecideAction] = []
            await runner.run(effect) { actions.append($0) }
            #expect(actions == [expected])
            service.failure = .cancel
            actions = []
            await runner.run(effect) { actions.append($0) }
            #expect(actions.isEmpty)
        }
    }

    @Test("Export succeeds with a valid confirmation and fails without publishing invalid JSON")
    func exporting() async {
        let service = FakeDecideService()
        let runner = DecideRunner(service: service)
        var actions: [DecideAction] = []
        let ticket = DecideExportTicket(id: 1, session: 1, revision: 2, presentation: 3)
        await runner.run(.export(ticket, service.output, [0: "yes"])) { actions.append($0) }
        if case .exportPrepared(let returned, let data) = actions.first {
            #expect(returned == ticket)
            #expect(!data.isEmpty)
        } else {
            Issue.record("Expected an export.")
        }
        actions = []
        await runner.run(.export(ticket, service.output, [0: "absent"])) { actions.append($0) }
        #expect(actions == [.exportFailed(ticket, "A reviewer confirmation is not an allowed option.")])
    }

    @Test("The store sends cancellation, waits for the service and can run again after a late completion")
    func stopAndRunAgain() async {
        let service = FakeDecideService()
        service.suspendDecision = true
        let runner = DecideRunner(service: service)
        let completion = EffectCompletion()
        var initial = DecideState(input: service.input)
        initial.model = service.info
        let store = Store<DecideFeature>(initialState: initial) { effect, send in
            await runner.run(effect, send: send)
            completion.finished(effect)
        }
        store.send(.run(reuse: true))
        await service.waitForStart()
        store.send(.stop)
        await service.waitForStop()
        #expect(store.state.phase == .stopping)
        #expect(store.state.output == nil)
        service.complete()
        await completion.wait(for: .waitForIdle(2))
        #expect(service.sawCancellation)
        #expect(store.state.phase == .idle && store.state.output == nil)
        service.suspendDecision = false
        store.send(.run(reuse: true))
        await completion.wait(for: .decide(3, 0, service.input, reuse: true))
        #expect(store.state.output == service.output)
        #expect(store.state.phase == .idle)
        store.send(.ended)
    }
}

@MainActor
private final class FakeDecideService: DecideService {
    enum Failure { case error, cancel }
    struct TestError: Error, LocalizedError {
        var errorDescription: String? { "Expected failure." }
    }

    var failure: Failure?
    var suspendDecision = false
    var sawCancellation = false
    private var active = false
    private var started = false
    private var stopStarted = false
    private var decision: CheckedContinuation<Void, Never>?
    private var startWaiter: CheckedContinuation<Void, Never>?
    private var stopWaiter: CheckedContinuation<Void, Never>?
    private var idleWaiters: [CheckedContinuation<Void, Never>] = []
    let info = DecideBundleInfo(name: "fixture", path: "/bundle", tokenLimit: 9_216)
    let input = DecideInput(document: "Document", items: [.blank(id: 0)])

    var output: DecideOutput {
        DecideOutput(
            rows: [
                DecideRow(
                    id: 0, key: "q", candidate: "yes",
                    probabilities: [
                        DecideProbability(id: 0, option: "yes", label: "Yes", probability: 0.9)
                    ], reviewFirst: false)
            ],
            modelName: "fixture", modelVersion: "v1", inputTokens: 42, reusedTokens: 20,
            seconds: 0.5, reusedDocument: true, recomputedAll: true,
            requestJSON: "{}", responseJSON: "{}")
    }

    var example: DecideExample {
        DecideExample(id: "fixture", title: "Fixture", input: input, extraItem: .blank(id: 1), searchHint: "Document")
    }

    func examples() async throws -> [DecideExample] {
        try checkFailure()
        return [example]
    }

    func load(path: String) async throws -> DecideBundleInfo {
        try checkFailure()
        return info
    }

    func count(_ input: DecideInput) async throws -> Int {
        try checkFailure()
        return 42
    }

    func read(_ url: URL) async throws -> String {
        try checkFailure()
        return "Imported document"
    }

    func decide(_ input: DecideInput, reuse: Bool) async throws -> DecideOutput {
        try checkFailure()
        active = true
        defer {
            active = false
            for waiter in idleWaiters { waiter.resume() }
            idleWaiters = []
        }
        started = true
        startWaiter?.resume()
        startWaiter = nil
        if suspendDecision { await withCheckedContinuation { decision = $0 } }
        sawCancellation = Task.isCancelled
        return output
    }

    func waitForIdle() async {
        stopStarted = true
        stopWaiter?.resume()
        stopWaiter = nil
        if active { await withCheckedContinuation { idleWaiters.append($0) } }
    }

    func waitForStart() async {
        if !started { await withCheckedContinuation { startWaiter = $0 } }
    }

    func waitForStop() async {
        if !stopStarted { await withCheckedContinuation { stopWaiter = $0 } }
    }

    func complete() {
        decision?.resume()
        decision = nil
    }

    private func checkFailure() throws {
        switch failure {
        case .error: throw TestError()
        case .cancel: throw CancellationError()
        case nil: break
        }
    }
}

@MainActor
private final class EffectCompletion {
    private var completed: [DecideEffect] = []
    private var waiting: [(DecideEffect, CheckedContinuation<Void, Never>)] = []

    func finished(_ effect: DecideEffect) {
        completed.append(effect)
        let ready = waiting.filter { $0.0 == effect }
        waiting.removeAll { $0.0 == effect }
        for value in ready { value.1.resume() }
    }

    func wait(for effect: DecideEffect) async {
        if !completed.contains(effect) { await withCheckedContinuation { waiting.append((effect, $0)) } }
    }
}
