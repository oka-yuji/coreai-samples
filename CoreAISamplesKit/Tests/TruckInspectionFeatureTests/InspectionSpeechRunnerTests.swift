import Foundation
import Testing
import TruckInspectionCore
import TruckInspectionFeature
import TruckInspectionRuntime

@MainActor
private final class InspectionSpeechFake: TruckInspectionSpeechService {
    var events: [InspectionSpeechEvent] = []
    var failure = false
    var timedOut = false
    var hold = false
    var ends = 0
    var finishes = 0
    var stops = 0
    var continuation: AsyncStream<InspectionSpeechEvent>.Continuation?
    let started = AsyncStream<Void>.makeStream()

    func prepare() async throws -> String {
        if failure { throw InspectionSpeechError.permissionDenied }
        return "ready"
    }

    func speak(_ text: String) async throws {
        if failure { throw InspectionSpeechError.noJapaneseVoice }
        if timedOut { throw InspectionSpeechError.playbackTimedOut }
    }

    func startListening(hints: [String]) async throws -> AsyncStream<InspectionSpeechEvent> {
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
    func stop() async -> String? {
        stops += 1
        continuation?.finish()
        return nil
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
