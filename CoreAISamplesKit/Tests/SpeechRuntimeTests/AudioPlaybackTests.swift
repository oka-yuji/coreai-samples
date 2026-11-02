import Foundation
import SpeechCore
import Testing

@testable import SpeechRuntime

@MainActor
final class FakePlaybackDevice: PlaybackDevice {
    var completion: ((Bool) -> Void)?
    var playResult = true
    var immediateCompletion: Bool?
    var holdsStart = false
    var pendingStart: (@MainActor (Bool) -> Void)?
    var holdsStop = false
    var pendingStop: (@MainActor () -> Void)?
    var stopCount = 0
    var playCount = 0
    let started = TestSignal()
    func play(started report: @escaping @MainActor (Bool) -> Void) {
        playCount += 1
        started.fire()
        if let immediateCompletion { completion?(immediateCompletion) }
        if holdsStart { pendingStart = report } else { report(playResult) }
    }
    func stop(completed: @escaping @MainActor () -> Void) {
        stopCount += 1
        if holdsStop { pendingStop = completed } else { completed() }
    }
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

    @Test(
        "A stop while the device is still starting ends playback once and ignores the late start result",
        arguments: [true, false])
    func stopWhileStarting(lateResult: Bool) async throws {
        let device = FakePlaybackDevice()
        device.holdsStart = true
        let playback = AudioPlayback { _ in device }
        let task = Task { try await playback.play(Data()) }
        defer {
            playback.stop()
            task.cancel()
        }
        try await device.started.wait()
        playback.stop()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(device.stopCount == 1 && device.completion == nil)
        let report = try #require(device.pendingStart)
        report(lateResult)
        #expect(device.stopCount == 1 && device.playCount == 1)
    }

    @Test("The session is not deactivated until the device has finished stopping")
    func sessionWaitsForDeviceStop() async throws {
        let session = FakePlaybackSession()
        let device = FakePlaybackDevice()
        device.holdsStop = true
        let playback = AudioPlayback(session: session) { _ in device }
        var returned = false
        let task = Task {
            defer { returned = true }
            try await playback.play(Data())
        }
        defer {
            device.pendingStop?()
            task.cancel()
        }
        try await device.started.wait()
        playback.stop()
        try #require(device.stopCount == 1 && session.stops == 0)
        await #expect(throws: SpeechError.busy) { try await playback.play(Data()) }
        #expect(!returned)
        let stopped = try #require(device.pendingStop)
        device.pendingStop = nil
        stopped()
        #expect(session.stops == 1)
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(returned && device.stopCount == 1 && session.stops == 1)
    }

    @Test("A late start result from stopped playback cannot affect newer playback")
    func staleStartResult() async throws {
        let first = FakePlaybackDevice()
        let second = FakePlaybackDevice()
        first.holdsStart = true
        var devices = [first, second]
        let playback = AudioPlayback { _ in devices.removeFirst() }
        let old = Task { try await playback.play(Data()) }
        defer {
            playback.stop()
            old.cancel()
        }
        try await first.started.wait()
        playback.stop()
        await #expect(throws: CancellationError.self) { try await old.value }
        let current = Task { try await playback.play(Data()) }
        defer {
            playback.stop()
            current.cancel()
        }
        try await second.started.wait()
        let report = try #require(first.pendingStart)
        report(false)
        try #require(second.stopCount == 0 && second.completion != nil)
        second.completion?(true)
        try await current.value
        #expect(first.stopCount == 1 && second.stopCount == 1)
    }

    @Test("Stop waiters return only after the device stop and the session deactivation complete")
    func stopWaitersWaitForCleanup() async throws {
        let device = FakeSessionDevice()
        let player = FakePlaybackDevice()
        player.holdsStop = true
        let playback = AudioPlayback(session: ManagedPlaybackSession(device: device)) { _ in player }
        var returned = false
        var released = 0
        let playing = Task {
            defer { returned = true }
            try await playback.play(Data())
        }
        defer {
            player.pendingStop?()
            device.completed?()
            playing.cancel()
        }
        try await device.activationStarted.wait()
        device.ready?(.success(()))
        try await player.started.wait()
        playback.stop()
        let waiting = [TestSignal(), TestSignal()]
        let waiters = waiting.map { signal in
            Task {
                signal.fire()
                await playback.stopAndWait()
                released += 1
            }
        }
        for signal in waiting { try await signal.wait() }
        #expect(player.stopCount == 1 && device.deactivations == 0 && !returned && released == 0)
        let stopped = try #require(player.pendingStop)
        player.pendingStop = nil
        stopped()
        try await device.deactivationStarted.wait()
        #expect(device.deactivations == 1 && !returned && released == 0)
        device.completed?()
        await #expect(throws: CancellationError.self) { try await playing.value }
        for waiter in waiters { await waiter.value }
        #expect(returned && released == 2 && device.deactivations == 1 && player.stopCount == 1)
    }

    @Test("A stop with nothing playing does not block the next playback")
    func idleStop() async throws {
        let device = FakePlaybackDevice()
        device.immediateCompletion = true
        let playback = AudioPlayback { _ in device }
        playback.stop()
        let task = Task { try await playback.play(Data()) }
        defer {
            playback.stop()
            task.cancel()
        }
        try await device.started.wait()
        try await task.value
        #expect(device.playCount == 1 && device.stopCount == 1 && device.completion == nil)
    }
}
