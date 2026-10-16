import Foundation
import SpeechCore
import Testing

@testable import SpeechRuntime

@MainActor
final class FakePlaybackDevice: PlaybackDevice {
    var completion: ((Bool) -> Void)?
    var playResult = true
    var immediateCompletion: Bool?
    var stopCount = 0
    var playCount = 0
    let started = TestSignal()
    func play() -> Bool {
        playCount += 1
        started.fire()
        if let immediateCompletion { completion?(immediateCompletion) }
        return playResult
    }
    func stop() { stopCount += 1 }
}

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct AudioPlaybackTests {
    @Test("Synchronous completion and a refused start both resolve exactly once")
    func synchronousResults() async throws {
        let device = FakePlaybackDevice()
        device.immediateCompletion = true
        let playback = AudioPlayback { _ in device }
        try await playback.play(Data())
        #expect(device.stopCount == 1 && device.completion == nil)
        device.immediateCompletion = nil
        device.playResult = false
        await #expect(throws: SpeechError.playbackFailed) { try await playback.play(Data()) }
        #expect(device.stopCount == 2 && device.completion == nil)
    }

    @Test("Cancellation ends playback without a delegate callback")
    func taskCancellation() async throws {
        let device = FakePlaybackDevice()
        let playback = AudioPlayback { _ in device }
        let task = Task { try await playback.play(Data()) }
        defer {
            playback.stop()
            task.cancel()
        }
        try await device.started.wait()
        await #expect(throws: SpeechError.busy) { try await playback.play(Data()) }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(device.stopCount == 1 && device.completion == nil)
    }

    @Test("Late callbacks from stopped audio cannot finish newer playback")
    func staleCallback() async throws {
        let first = FakePlaybackDevice()
        let second = FakePlaybackDevice()
        var devices = [first, second]
        let playback = AudioPlayback { _ in devices.removeFirst() }
        let old = Task { try await playback.play(Data()) }
        defer {
            playback.stop()
            old.cancel()
        }
        try await first.started.wait()
        let callback = try #require(first.completion)
        old.cancel()
        playback.stop()
        await #expect(throws: CancellationError.self) { try await old.value }
        let current = Task { try await playback.play(Data()) }
        defer {
            playback.stop()
            current.cancel()
        }
        try await second.started.wait()
        callback(true)
        callback(false)
        #expect(second.stopCount == 0)
        second.completion?(true)
        try await current.value
        #expect(first.stopCount == 1 && second.stopCount == 1)
    }

    @Test("A cancelled caller does not create or start a player")
    func cancelledBeforeStart() async {
        var created = false
        let playback = AudioPlayback { _ in
            created = true
            return FakePlaybackDevice()
        }
        let gate = TestGate()
        let task = Task {
            await gate.wait()
            try await playback.play(Data())
        }
        task.cancel()
        gate.open()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(!created)
    }
}
