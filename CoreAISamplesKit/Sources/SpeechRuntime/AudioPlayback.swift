#if os(macOS) || os(iOS)
    import AVFAudio
    import Foundation
    import SpeechCore
    import Synchronization

    @MainActor
    protocol PlaybackDevice: AnyObject {
        var completion: ((Bool) -> Void)? { get set }
        func play(started: @escaping @MainActor (Bool) -> Void)
        func stop(completed: @escaping @MainActor () -> Void)
    }

    actor PlayerHardware {
        nonisolated private let queue = DispatchSerialQueue(label: "PlayerHardware")
        nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }
        private let player: AVAudioPlayer
        private var stopped = false

        init(data: Data) throws { player = try AVAudioPlayer(data: data) }
        func play(delegate: PlayerDevice) -> Bool {
            guard !stopped else { return false }
            player.delegate = delegate
            return player.prepareToPlay() && player.play()
        }
        func stop() {
            stopped = true
            player.delegate = nil
            player.stop()
        }
        nonisolated func play(delegate: PlayerDevice, started: @escaping @MainActor (Bool) -> Void) {
            queue.async {
                let result = self.assumeIsolated { $0.play(delegate: delegate) }
                DispatchQueue.main.async { started(result) }
            }
        }
        nonisolated func stop(completed: @escaping @MainActor () -> Void) {
            queue.async {
                self.assumeIsolated { $0.stop() }
                DispatchQueue.main.async { completed() }
            }
        }
    }

    @MainActor
    final class PlayerDevice: NSObject, PlaybackDevice, AVAudioPlayerDelegate {
        private let hardware: PlayerHardware
        var completion: ((Bool) -> Void)?

        init(data: Data) throws {
            hardware = try PlayerHardware(data: data)
            super.init()
        }
        func play(started: @escaping @MainActor (Bool) -> Void) { hardware.play(delegate: self, started: started) }
        func stop(completed: @escaping @MainActor () -> Void) { hardware.stop(completed: completed) }
        func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) { completion?(flag) }
        func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: (any Error)?) { completion?(false) }
    }

    @MainActor
    final class AudioPlayback {
        private let makeDevice: (Data) throws -> any PlaybackDevice
        private let session: any PlaybackSession
        private var device: (any PlaybackDevice)?
        private var continuation: CheckedContinuation<Void, any Error>?
        private var generation = 0
        private var finishing = false
        private var stopWaiters: [CheckedContinuation<Void, Never>] = []

        init(
            session: (any PlaybackSession)? = nil,
            makeDevice: @escaping (Data) throws -> any PlaybackDevice = { try PlayerDevice(data: $0) }
        ) {
            self.makeDevice = makeDevice
            self.session = session ?? PlaybackSessions.make()
        }

        func play(_ data: Data) async throws {
            guard continuation == nil else { throw SpeechError.busy }
            try Task.checkCancellation()
            generation += 1
            let ticket = generation
            let created = try makeDevice(data)
            let cancelled = Mutex(false)
            try await withTaskCancellationHandler {
                try Task.checkCancellation()
                try await withCheckedThrowingContinuation { pending in
                    continuation = pending
                    device = created
                    created.completion = { [weak self] success in
                        self?.finish(ticket, result: success ? .success(()) : .failure(SpeechError.playbackFailed))
                    }
                    session.start(
                        interrupted: { [weak self] in
                            self?.finish(ticket, result: .failure(CancellationError()))
                        },
                        ready: { [weak self] result in
                            guard let self, continuation != nil, generation == ticket, !finishing else { return }
                            if cancelled.withLock({ $0 }) {
                                finish(ticket, result: .failure(CancellationError()))
                                return
                            }
                            switch result {
                            case .failure(let error): finish(ticket, result: .failure(error))
                            case .success:
                                created.play { [weak self] started in
                                    guard let self, !started else { return }
                                    finish(ticket, result: .failure(SpeechError.playbackFailed))
                                }
                            }
                        })
                }
            } onCancel: {
                cancelled.withLock { $0 = true }
                DispatchQueue.main.async { [weak self] in
                    self?.finish(ticket, result: .failure(CancellationError()))
                }
            }
        }

        func stop() { finish(generation, result: .failure(CancellationError())) }

        func stopAndWait() async {
            stop()
            if continuation != nil {
                await withCheckedContinuation { stopWaiters.append($0) }
            }
        }

        private func finish(_ ticket: Int, result: Result<Void, any Error>) {
            guard ticket == generation, continuation != nil, !finishing else { return }
            finishing = true
            device?.completion = nil
            let stopped: @MainActor () -> Void = { [weak self] in
                self?.session.stop { [weak self] in self?.complete(ticket, result: result) }
            }
            if let device { device.stop(completed: stopped) } else { stopped() }
        }
        private func complete(_ ticket: Int, result: Result<Void, any Error>) {
            guard ticket == generation, let pending = continuation, finishing else { return }
            continuation = nil
            device = nil
            finishing = false
            let waiters = stopWaiters
            stopWaiters.removeAll()
            pending.resume(with: result)
            for waiter in waiters { waiter.resume() }
        }
    }
#endif
