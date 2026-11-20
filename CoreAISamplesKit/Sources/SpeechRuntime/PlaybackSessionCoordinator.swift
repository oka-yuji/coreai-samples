#if os(macOS) || os(iOS)
    import Foundation
    import SpeechCore

    @MainActor
    final class PlaybackSessionCoordinator {
        private typealias Request = (
            owner: UUID, interrupted: @MainActor () -> Void, ready: @MainActor (Result<Void, any Error>) -> Void
        )

        private let device: any PlaybackSessionDevice
        private var current: (owner: UUID, session: ManagedPlaybackSession, stopping: Bool)?
        private var waiting: Request?

        init(device: any PlaybackSessionDevice) { self.device = device }

        func start(
            owner: UUID, interrupted: @escaping @MainActor () -> Void,
            ready: @escaping @MainActor (Result<Void, any Error>) -> Void
        ) {
            let request: Request = (owner, interrupted, ready)
            guard let current else {
                begin(request)
                return
            }
            guard current.stopping, current.owner != owner, waiting == nil else {
                ready(.failure(SpeechError.busy))
                return
            }
            waiting = request
        }

        func stop(owner: UUID, completed: @escaping @MainActor () -> Void) {
            if let waiting, waiting.owner == owner {
                self.waiting = nil
                waiting.ready(.failure(CancellationError()))
                completed()
                return
            }
            guard let active = current, active.owner == owner else {
                completed()
                return
            }
            current?.stopping = true
            let session = active.session
            session.stop { [weak self, weak session] in
                if let self, let session, current?.session === session {
                    current = nil
                    if let waiting {
                        self.waiting = nil
                        begin(waiting)
                    }
                }
                completed()
            }
        }

        private func begin(_ request: Request) {
            let session = ManagedPlaybackSession(device: device)
            current = (request.owner, session, false)
            session.start(
                interrupted: { [weak self, weak session] in
                    guard let session, let active = self?.current, active.session === session, !active.stopping else {
                        return
                    }
                    request.interrupted()
                },
                ready: { [weak self, weak session] result in
                    guard let self, let session, let active = current, active.session === session else { return }
                    if active.stopping {
                        request.ready(.failure(CancellationError()))
                        return
                    }
                    if case .failure = result { current = nil }
                    request.ready(result)
                })
        }
    }
#endif
