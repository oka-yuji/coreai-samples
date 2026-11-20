import SpeechCore
import Testing

@testable import SpeechRuntime

@MainActor
struct PlaybackSessionCoordinatorTests {
    @Test("A new owner waits for both the old activation and its deactivation")
    func delayedHandoff() throws {
        let device = FakeSessionDevice()
        let coordinator = PlaybackSessionCoordinator(device: device)
        let old = PlaybackSessionOwner(coordinator: coordinator)
        let next = PlaybackSessionOwner(coordinator: coordinator)
        var previous: Result<Void, any Error>?
        var current: Result<Void, any Error>?
        var stops: [Int] = []
        old.start(interrupted: {}, ready: { previous = $0 })
        let oldActivation = try #require(device.ready)
        old.stop { stops.append(1) }
        old.stop { stops.append(2) }
        next.start(interrupted: {}, ready: { current = $0 })
        #expect(device.activations == 1 && device.deactivations == 0 && stops.isEmpty)
        oldActivation(.success(()))
        let cancelled = try #require(previous)
        #expect(throws: CancellationError.self) { try cancelled.get() }
        #expect(device.activations == 1 && device.deactivations == 1 && stops.isEmpty)
        let oldDeactivation = try #require(device.completed)
        oldDeactivation()
        #expect(device.activations == 2 && stops == [1, 2] && current == nil)
        let newActivation = try #require(device.ready)
        newActivation(.success(()))
        try #require(current).get()
        oldActivation(.failure(SpeechError.playbackFailed))
        oldDeactivation()
        old.stop { stops.append(3) }
        #expect(device.activations == 2 && device.deactivations == 1 && stops == [1, 2, 3])
        next.stop { stops.append(4) }
        #expect(device.deactivations == 2 && stops == [1, 2, 3])
        device.completed?()
        #expect(stops == [1, 2, 3, 4])
    }

    @Test("A failed activation releases a waiting owner without deactivating the device")
    func failedActivation() throws {
        let device = FakeSessionDevice()
        let coordinator = PlaybackSessionCoordinator(device: device)
        let old = PlaybackSessionOwner(coordinator: coordinator)
        let next = PlaybackSessionOwner(coordinator: coordinator)
        var stops = 0
        var nextReady = 0
        old.start(interrupted: {}, ready: { _ in })
        old.stop { stops += 1 }
        next.start(interrupted: {}, ready: { _ in nextReady += 1 })
        let failed = try #require(device.ready)
        failed(.failure(SpeechError.playbackFailed))
        #expect(stops == 1 && device.activations == 2 && device.deactivations == 0 && nextReady == 0)
        failed(.success(()))
        device.ready?(.success(()))
        old.stop { stops += 1 }
        #expect(stops == 2 && nextReady == 1 && device.deactivations == 0)
        next.stop { stops += 1 }
        device.completed?()
        #expect(stops == 3)
    }

    @Test("An owner that stops while waiting never activates and its late stop leaves the next owner alone")
    func cancelledWaiter() throws {
        let device = FakeSessionDevice()
        let coordinator = PlaybackSessionCoordinator(device: device)
        let active = PlaybackSessionOwner(coordinator: coordinator)
        let waiting = PlaybackSessionOwner(coordinator: coordinator)
        let replacement = PlaybackSessionOwner(coordinator: coordinator)
        var attempts: [Result<Void, any Error>] = []
        var stops = 0
        var replacementReady = false
        active.start(interrupted: {}, ready: { _ in })
        device.ready?(.success(()))
        waiting.start(interrupted: {}, ready: { attempts.append($0) })
        let busy = try #require(attempts.first)
        #expect(throws: SpeechError.busy) { try busy.get() }
        waiting.stop { stops += 1 }
        #expect(stops == 1 && device.deactivations == 0)
        active.stop { stops += 1 }
        waiting.start(interrupted: {}, ready: { attempts.append($0) })
        #expect(attempts.count == 1 && device.activations == 1)
        waiting.stop { stops += 1 }
        let cancelled = try #require(attempts.last)
        #expect(throws: CancellationError.self) { try cancelled.get() }
        replacement.start(interrupted: {}, ready: { _ in replacementReady = true })
        waiting.stop { stops += 1 }
        #expect(stops == 3 && attempts.count == 2 && device.activations == 1)
        device.completed?()
        #expect(stops == 4 && device.activations == 2 && !replacementReady)
        device.ready?(.success(()))
        waiting.stop { stops += 1 }
        #expect(stops == 5 && attempts.count == 2 && replacementReady && device.deactivations == 1)
        replacement.stop { stops += 1 }
        device.completed?()
        #expect(stops == 6)
    }
}
