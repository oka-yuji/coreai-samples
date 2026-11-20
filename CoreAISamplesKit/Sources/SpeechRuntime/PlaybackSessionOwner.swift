#if os(macOS) || os(iOS)
    import Foundation

    @MainActor
    final class PlaybackSessionOwner: PlaybackSession {
        private let id = UUID()
        private let coordinator: PlaybackSessionCoordinator

        init(coordinator: PlaybackSessionCoordinator) {
            self.coordinator = coordinator
        }

        func start(
            interrupted: @escaping @MainActor () -> Void,
            ready: @escaping @MainActor (Result<Void, any Error>) -> Void
        ) {
            coordinator.start(owner: id, interrupted: interrupted, ready: ready)
        }

        func stop(completed: @escaping @MainActor () -> Void) {
            coordinator.stop(owner: id, completed: completed)
        }
    }
#endif
