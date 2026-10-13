#if os(macOS) || os(iOS)
    import Foundation
    import SpeechCore

    @MainActor
    protocol PlaybackSession {
        func start(
            interrupted: @escaping @MainActor () -> Void,
            ready: @escaping @MainActor (Result<Void, any Error>) -> Void)
        func stop(completed: @escaping @MainActor () -> Void)
    }

    @MainActor
    protocol PlaybackSessionDevice {
        func activate(
            interrupted: @escaping @MainActor () -> Void,
            ready: @escaping @MainActor (Result<Void, any Error>) -> Void)
        func deactivate(completed: @escaping @MainActor () -> Void)
    }

    @MainActor
    final class ManagedPlaybackSession: PlaybackSession {
        private enum Phase { case idle, starting, active, stopping }
        private let device: any PlaybackSessionDevice
        private var phase: Phase = .idle
        private var generation = 0
        private var stops: [@MainActor () -> Void] = []
        init(device: any PlaybackSessionDevice) { self.device = device }

        func start(
            interrupted: @escaping @MainActor () -> Void,
            ready: @escaping @MainActor (Result<Void, any Error>) -> Void
        ) {
            guard phase == .idle else {
                ready(.failure(SpeechError.busy))
                return
            }
            generation += 1
            let ticket = generation
            phase = .starting
            device.activate(interrupted: interrupted) { [weak self] result in
                guard let self, generation == ticket, phase == .starting else { return }
                switch result {
                case .success:
                    phase = .active
                    ready(result)
                    if !stops.isEmpty { deactivate() }
                case .failure:
                    phase = .idle
                    ready(result)
                    completeStops()
                }
            }
        }
        func stop(completed: @escaping @MainActor () -> Void) {
            stops.append(completed)
            switch phase {
            case .idle: completeStops()
            case .starting, .stopping: break
            case .active: deactivate()
            }
        }
        private func deactivate() {
            guard phase == .active else { return }
            phase = .stopping
            let ticket = generation
            device.deactivate { [weak self] in
                guard let self, generation == ticket, phase == .stopping else { return }
                phase = .idle
                completeStops()
            }
        }
        private func completeStops() {
            let pending = stops
            stops.removeAll()
            for completed in pending { completed() }
        }
    }

    @MainActor
    final class PassivePlaybackSession: PlaybackSession {
        func start(
            interrupted: @escaping @MainActor () -> Void,
            ready: @escaping @MainActor (Result<Void, any Error>) -> Void
        ) { ready(.success(())) }
        func stop(completed: @escaping @MainActor () -> Void) { completed() }
    }

    @MainActor
    enum PlaybackSessions {
        static func make() -> any PlaybackSession {
            #if os(iOS)
                ManagedPlaybackSession(device: PhonePlaybackSessionDevice())
            #else
                PassivePlaybackSession()
            #endif
        }
    }
#endif
