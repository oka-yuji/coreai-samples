import Foundation
import SpeechCore
import Testing

@testable import SpeechRuntime

@MainActor
final class FakeSessionDevice: PlaybackSessionDevice {
    let activationStarted = TestSignal()
    let deactivationStarted = TestSignal()
    var ready: (@MainActor (Result<Void, any Error>) -> Void)?
    var completed: (@MainActor () -> Void)?
    var activations = 0
    var deactivations = 0
    func activate(
        interrupted: @escaping @MainActor () -> Void,
        ready: @escaping @MainActor (Result<Void, any Error>) -> Void
    ) {
        activations += 1
        self.ready = ready
        activationStarted.fire()
    }
    func deactivate(completed: @escaping @MainActor () -> Void) {
        deactivations += 1
        self.completed = completed
        deactivationStarted.fire()
    }
}

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ManagedPlaybackSessionTests {
    @Test("Stopping during activation waits for activation and deactivation without starting audio")
    func cancelActivation() async throws {
        let device = FakeSessionDevice()
        let player = FakePlaybackDevice()
        let playback = AudioPlayback(session: ManagedPlaybackSession(device: device)) { _ in player }
        var returned = false
        let playing = Task {
            defer { returned = true }
            try await playback.play(Data())
        }
        defer {
            device.ready?(.failure(CancellationError()))
            device.completed?()
            playing.cancel()
        }
        try await device.activationStarted.wait()
        playing.cancel()
        playback.stop()
        #expect(device.deactivations == 0 && !returned)
        await #expect(throws: SpeechError.busy) { try await playback.play(Data()) }
        device.ready?(.success(()))
        try await device.deactivationStarted.wait()
        #expect(player.playCount == 0 && !returned)
        device.completed?()
        await #expect(throws: CancellationError.self) { try await playing.value }
        await playback.stopAndWait()
        #expect(returned && device.deactivations == 1)
    }

    @Test("A failed activation completes stop waiters without requesting deactivation")
    func failedActivation() async throws {
        let device = FakeSessionDevice()
        let player = FakePlaybackDevice()
        let playback = AudioPlayback(session: ManagedPlaybackSession(device: device)) { _ in player }
        let playing = Task { try await playback.play(Data()) }
        defer {
            device.ready?(.failure(CancellationError()))
            device.completed?()
            playing.cancel()
        }
        try await device.activationStarted.wait()
        playback.stop()
        let stopped = Task { await playback.stopAndWait() }
        device.ready?(.failure(SpeechError.playbackFailed))
        await #expect(throws: CancellationError.self) { try await playing.value }
        await stopped.value
        #expect(player.playCount == 0 && device.deactivations == 0)
    }

    @Test("Playback completion does not release its caller before session deactivation finishes")
    func completionWaitsForCleanup() async throws {
        let device = FakeSessionDevice()
        let player = FakePlaybackDevice()
        let playback = AudioPlayback(session: ManagedPlaybackSession(device: device)) { _ in player }
        var returned = false
        let playing = Task {
            try await playback.play(Data())
            returned = true
        }
        defer {
            device.ready?(.failure(CancellationError()))
            device.completed?()
            playing.cancel()
        }
        try await device.activationStarted.wait()
        device.ready?(.success(()))
        try await player.started.wait()
        player.completion?(true)
        try await device.deactivationStarted.wait()
        #expect(!returned && player.stopCount == 1)
        device.completed?()
        try await playing.value
        #expect(returned)
    }
}
