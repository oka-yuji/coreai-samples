import FeatureStore
import Foundation
import Testing
import TruckInspectionCore
import TruckInspectionFeature
import TruckInspectionRuntime

private struct DeactivationFailure: Error {}

@MainActor
private final class ExecutionGateFake {
    var open = false
}

@MainActor
private final class InspectionSpeechFake: TruckInspectionSpeechService {
    var events: [InspectionSpeechEvent] = []
    var failure = false
    var prepareFailure: InspectionSpeechError?
    var timedOut = false
    var hold = false
    var holdsPrepare = false
    var stopFailures = 0
    var prepares = 0
    var speaks = 0
    var listens = 0
    var ends = 0
    var finishes = 0
    var stops = 0
    var continuation: AsyncStream<InspectionSpeechEvent>.Continuation?
    var pendingPrepare: CheckedContinuation<Void, Never>?
    let started = AsyncStream<Void>.makeStream()
    let preparing = AsyncStream<Void>.makeStream()

    func prepare() async throws -> String {
        prepares += 1
        if holdsPrepare {
            preparing.continuation.yield(())
            await withCheckedContinuation { pendingPrepare = $0 }
        }
        if let prepareFailure { throw prepareFailure }
        if failure { throw InspectionSpeechError.routeUnavailable }
        return "ready"
    }

    func speak(_ text: String) async throws {
        speaks += 1
        if failure { throw InspectionSpeechError.noJapaneseVoice }
        if timedOut { throw InspectionSpeechError.playbackTimedOut }
    }

    func startListening(hints: [String]) async throws -> AsyncStream<InspectionSpeechEvent> {
        listens += 1
        if failure { throw InspectionSpeechError.unavailable }
        let pair = AsyncStream<InspectionSpeechEvent>.makeStream()
        continuation = pair.continuation
        for event in events { pair.continuation.yield(event) }
        if !hold { pair.continuation.finish() }
        started.continuation.yield(())
        return pair.stream
    }

    func endAudio() async { ends += 1 }
    func finishListening() async {
        finishes += 1
        continuation?.finish()
    }
    func stop() async throws {
        stops += 1
        continuation?.finish()
        guard stopFailures == 0 else {
            stopFailures -= 1
            throw DeactivationFailure()
        }
    }

    func finishPrepare() {
        let continuation = pendingPrepare
        pendingPrepare = nil
        continuation?.resume()
    }
}

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct InspectionSpeechRunnerTests {
    @Test("TV2: Speech setup and playback report success, failure and deadline expiry")
    func outcomes() async {
        let speech = InspectionSpeechFake()
        let runner = TruckInspectionRunner(model: InspectionModelFake(), speech: speech)
        var actions: [InspectionAction] = []
        await runner.run(.voice(1, .prepare)) { actions.append($0) }
        guard case .voice(.prepared(1, "ready")) = actions.last else {
            Issue.record("Expected setup result")
            return
        }
        await runner.run(.voice(2, .speak("question"))) { actions.append($0) }
        guard case .voice(.spoken(2, false)) = actions.last else {
            Issue.record("Expected playback completion")
            return
        }
        speech.timedOut = true
        await runner.run(.voice(3, .speak("question"))) { actions.append($0) }
        guard case .voice(.spoken(3, true)) = actions.last else {
            Issue.record("Expected bounded playback")
            return
        }
        speech.failure = true
        for effect: InspectionVoiceEffect in [.prepare, .speak("question"), .listen([])] {
            await runner.run(.voice(4, effect)) { actions.append($0) }
            guard case .voice(.failed(4, _)) = actions.last else {
                Issue.record("Expected voice failure")
                return
            }
        }
    }

    @Test("TV2: Empty final and final timeout preserve the partial answer and stop capture", arguments: [true, false])
    func finalFallback(emptyFinal: Bool) async {
        let speech = InspectionSpeechFake()
        speech.events = [
            .partial("last answer", elapsed: 1), .partial("", elapsed: 1.5), .level(-85, elapsed: 0.1),
            .level(-90, elapsed: 2.2),
        ]
        if emptyFinal { speech.events.append(.final("")) } else { speech.events += [.tick(2.3), .tick(4.4)] }
        let runner = TruckInspectionRunner(model: InspectionModelFake(), speech: speech)
        var actions: [InspectionAction] = []
        await runner.run(.voice(5, .listen(["hint"]))) { actions.append($0) }
        guard case .voice(.heard(5, "last answer")) = actions.last else {
            Issue.record("Expected partial rescue")
            return
        }
        #expect(speech.finishes == 1)
        #expect(speech.ends == (emptyFinal ? 0 : 1))
        #expect(
            actions.contains {
                if case .diagnostic(_, let text) = $0 { text.contains("very low input") } else { false }
            })
    }

    @Test("TV2: Recognition errors clean up and never produce an answer")
    func recognitionFailure() async {
        let speech = InspectionSpeechFake()
        speech.events = [.failure("recognizer failed")]
        let runner = TruckInspectionRunner(model: InspectionModelFake(), speech: speech)
        var actions: [InspectionAction] = []
        await runner.run(.voice(6, .listen([]))) { actions.append($0) }
        #expect(speech.finishes == 1)
        guard case .voice(.failed(6, _)) = actions.last else {
            Issue.record("Expected recognition failure")
            return
        }
    }

    @Test("INSPECT-01: A closed execution gate rejects every model and voice side effect before its service runs")
    func closedGate() async throws {
        let catalog = try InspectionCatalogLoader.load()
        let item = try #require(catalog.items.first)
        let model = InspectionModelFake()
        let speech = InspectionSpeechFake()
        let gate = ExecutionGateFake()
        let runner = TruckInspectionRunner(model: model, speech: speech, permits: { gate.open })
        var actions: [InspectionAction] = []
        let effects: [InspectionEffect] = [
            .prepare(1, URL(filePath: "/asset")), .classify(2, "answer", item, catalog),
            .measure(3, "answer", item, catalog),
        ]
        for effect in effects {
            await runner.run(effect) { actions.append($0) }
            guard case .failed = actions.last else {
                Issue.record("Expected a rejected model effect")
                return
            }
        }
        for voice: InspectionVoiceEffect in [.prepare, .speak("question"), .listen([])] {
            await runner.run(.voice(4, voice)) { actions.append($0) }
            guard case .voice(.failed(4, _)) = actions.last else {
                Issue.record("Expected a rejected voice effect")
                return
            }
        }
        #expect(model.preparedFolders.isEmpty)
        #expect(model.classifiedTexts.isEmpty)
        #expect(speech.prepares == 0 && speech.speaks == 0 && speech.listens == 0)
        gate.open = true
        await runner.run(.prepare(5, URL(filePath: "/asset"))) { actions.append($0) }
        guard case .modelLoaded(5, _) = actions.last else {
            Issue.record("Expected preparation after the gate opens")
            return
        }
    }

    @Test(
        "INSPECT-02: Denied permissions and missing Japanese recognition or voice report their limit instead of a generic failure",
        arguments: [
            (InspectionSpeechError.microphoneDenied, InspectionVoiceLimit.microphoneDenied),
            (.speechRecognitionDenied, .speechRecognitionDenied), (.unavailable, .recognitionUnavailable),
            (.noJapaneseVoice, .voiceUnavailable),
        ])
    func limits(error: InspectionSpeechError, limit: InspectionVoiceLimit) async {
        let speech = InspectionSpeechFake()
        speech.prepareFailure = error
        let runner = TruckInspectionRunner(model: InspectionModelFake(), speech: speech)
        var actions: [InspectionAction] = []
        await runner.run(.voice(1, .prepare)) { actions.append($0) }
        guard case .voice(.limited(1, let reported)) = actions.last else {
            Issue.record("Expected a voice limit")
            return
        }
        #expect(reported == limit)
        speech.prepareFailure = .routeUnavailable
        await runner.run(.voice(2, .prepare)) { actions.append($0) }
        guard case .voice(.failed(2, _)) = actions.last else {
            Issue.record("Expected a generic voice failure")
            return
        }
    }

    @Test("INSPECT-03: An audio deactivation failure reports a failed stop and a retry stops idempotently")
    func deactivationFailure() async throws {
        let model = InspectionModelFake()
        let speech = InspectionSpeechFake()
        speech.stopFailures = 3
        let runner = TruckInspectionRunner(model: model, speech: speech)
        var actions: [InspectionAction] = []
        await runner.run(.stop(1, unload: true)) { actions.append($0) }
        guard case .stopFailed(1, _) = actions.last else {
            Issue.record("Expected a failed stop")
            return
        }
        await runner.run(.stop(2, unload: false)) { actions.append($0) }
        guard case .stopFailed(2, _) = actions.last else {
            Issue.record("Expected a failed stop without unloading")
            return
        }
        #expect(!actions.contains { if case .stopped = $0 { true } else { false } })
        await #expect(throws: InspectionReleaseError.self) { try await runner.release() }
        #expect(model.unloads == 0)
        await runner.run(.stop(3, unload: true)) { actions.append($0) }
        guard case .stopped(3, true) = actions.last else {
            Issue.record("Expected a stop after the retry")
            return
        }
        try await runner.release()
        #expect(speech.stops == 5)
        #expect(model.unloads == 2)
    }

    @Test("INSPECT-03: Release waits for a pending permission prompt and finishes even when its caller is cancelled")
    func releaseJoinsPermission() async throws {
        let model = InspectionModelFake()
        let speech = InspectionSpeechFake()
        speech.holdsPrepare = true
        let runner = TruckInspectionRunner(model: model, speech: speech)
        var actions: [InspectionAction] = []
        let operation = Task { await runner.run(.voice(1, .prepare)) { actions.append($0) } }
        var prompts = speech.preparing.stream.makeAsyncIterator()
        _ = await prompts.next()
        operation.cancel()
        var released = false
        let release = Task {
            try await runner.release()
            released = true
        }
        release.cancel()
        var drains = model.draining.stream.makeAsyncIterator()
        _ = await drains.next()
        #expect(!released)
        #expect(model.unloads == 0)
        speech.finishPrepare()
        try await release.value
        await operation.value
        #expect(released)
        #expect(model.unloads == 1)
        #expect(speech.stops == 1)
        #expect(!actions.contains { if case .voice(.prepared) = $0 { true } else { false } })
    }

    @Test("INSPECT-03: Closing a managed store and awaiting release stops listening and releases the selected model")
    func managedClose() async throws {
        let folder = URL(filePath: "/library/artifacts/gliner/revision")
        let model = InspectionModelFake()
        let speech = InspectionSpeechFake()
        speech.hold = true
        let runner = TruckInspectionRunner(model: model, speech: speech)
        let store = Store<TruckInspectionFeature>(
            initialState: InspectionState(
                catalog: try InspectionCatalogLoader.load(), modelFolder: folder, execution: .managed,
                artifact: "gliner2.5-multi/variant@revision"),
            run: runner.run)
        store.send(.appeared)
        #expect(model.preparedFolders.isEmpty)
        #expect(speech.prepares == 0)
        store.send(.voice(.start))
        var listening = speech.started.stream.makeAsyncIterator()
        _ = await listening.next()
        #expect(model.preparedFolders == [folder])
        #expect(store.state.phase == .listening)
        store.send(.closed)
        try await runner.release()
        #expect(!store.state.active)
        #expect(speech.stops >= 1)
        #expect(speech.finishes >= 1)
        #expect(model.unloads >= 1)
        #expect(!store.state.voiceActive)
    }

    @Test("TV2: Cancellation drains the listening operation before reporting stopped")
    func cancellation() async {
        let speech = InspectionSpeechFake()
        speech.hold = true
        let runner = TruckInspectionRunner(model: InspectionModelFake(), speech: speech)
        var actions: [InspectionAction] = []
        let operation = Task { await runner.run(.voice(7, .listen([]))) { actions.append($0) } }
        var started = speech.started.stream.makeAsyncIterator()
        _ = await started.next()
        operation.cancel()
        await runner.run(.stop(8, unload: false)) { actions.append($0) }
        await operation.value
        #expect(speech.stops == 1)
        #expect(speech.finishes >= 1)
        #expect(!actions.contains { if case .voice(.heard) = $0 { true } else { false } })
        guard case .stopped(8, false) = actions.last else {
            Issue.record("Expected cleanup completion")
            return
        }
    }
}
