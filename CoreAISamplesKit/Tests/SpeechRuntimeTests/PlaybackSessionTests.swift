import CoreAI
import Foundation
import SpeechCore
import SpeechFeature
import Testing

@testable import SpeechRuntime

@MainActor
final class FakePlaybackSession: PlaybackSession {
    var fail = false
    var interruptImmediately = false
    var interruptions: [@MainActor () -> Void] = []
    var starts = 0
    var stops = 0
    func start(
        interrupted: @escaping @MainActor () -> Void,
        ready: @escaping @MainActor (Result<Void, any Error>) -> Void
    ) {
        starts += 1
        interruptions.append(interrupted)
        if fail {
            ready(.failure(SpeechError.playbackFailed))
            return
        }
        if interruptImmediately { interrupted() }
        ready(.success(()))
    }
    func stop(completed: @escaping @MainActor () -> Void) {
        stops += 1
        completed()
    }
}

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct PlaybackSessionTests {
    @Test("Session setup failure and immediate interruption never start a player")
    func setupFailure() async {
        let session = FakePlaybackSession()
        let device = FakePlaybackDevice()
        let playback = AudioPlayback(session: session) { _ in device }
        session.fail = true
        await #expect(throws: SpeechError.playbackFailed) { try await playback.play(Data()) }
        #expect(session.stops == 1 && device.stopCount == 1)
        session.fail = false
        session.interruptImmediately = true
        await #expect(throws: CancellationError.self) { try await playback.play(Data()) }
        #expect(session.stops == 2 && device.stopCount == 2 && device.playCount == 0)
    }

    @Test("A past session interruption cannot stop a subsequent player")
    func staleInterruption() async throws {
        let session = FakePlaybackSession()
        let first = FakePlaybackDevice()
        let second = FakePlaybackDevice()
        var devices = [first, second]
        let playback = AudioPlayback(session: session) { _ in devices.removeFirst() }
        let old = Task { try await playback.play(Data()) }
        defer {
            playback.stop()
            old.cancel()
        }
        try await first.started.wait()
        let callback = try #require(session.interruptions.first)
        callback()
        await #expect(throws: CancellationError.self) { try await old.value }
        let current = Task { try await playback.play(Data()) }
        defer {
            playback.stop()
            current.cancel()
        }
        try await second.started.wait()
        callback()
        #expect(second.stopCount == 0 && session.stops == 1)
        second.completion?(true)
        try await current.value
        #expect(session.starts == 2 && session.stops == 2)
    }

    @Test("An external interruption returns the current speech screen to an idle state")
    func interruptionState() async throws {
        let service = FakeSpeechService()
        service.interruptPlayback = true
        let runner = SpeechRunner(service: service)
        var state = SpeechState()
        state.presented = true
        state.audio = service.audio
        _ = SpeechFeature.reduce(&state, .play)
        let session = state.session
        _ = SpeechFeature.reduce(&state, .playbackInterrupted(session - 1))
        #expect(state.phase == .playing)
        await runner.run(.play(session, service.audio.wav)) { _ = SpeechFeature.reduce(&state, $0) }
        #expect(state.phase == .idle && state.canPlay && state.audio == service.audio)
    }

    @Test("Static phone stages avoid reshape specialization while Mac keeps its verified settings")
    func stageOptions() {
        #expect(!Stage.options(dynamicShape: false, onPhone: true).expectFrequentReshapes)
        #expect(Stage.options(dynamicShape: true, onPhone: true).expectFrequentReshapes)
        #expect(Stage.options(dynamicShape: false, onPhone: false).expectFrequentReshapes)
        #expect(Stage.options(dynamicShape: true, onPhone: false).preferredComputeUnitKind == .gpu)
    }
}
