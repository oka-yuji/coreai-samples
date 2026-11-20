import SpeechRuntime

@MainActor
final class SpokenPlaybackProbe: PlaybackSession {
    private(set) var isActive = false
    private(set) var starts = 0
    private var waiters: [@MainActor () -> Void] = []
    private var observed: CheckedContinuation<Void, Never>?

    func start(
        interrupted: @escaping @MainActor () -> Void,
        ready: @escaping @MainActor (Result<Void, any Error>) -> Void
    ) {
        starts += 1
        isActive = true
        ready(.success(()))
    }

    func stop(completed: @escaping @MainActor () -> Void) {
        guard isActive else {
            completed()
            return
        }
        waiters.append(completed)
        observed?.resume()
        observed = nil
    }

    func waitForDeactivation() async {
        if waiters.isEmpty { await withCheckedContinuation { observed = $0 } }
    }

    func acknowledge() {
        isActive = false
        let ready = waiters
        waiters.removeAll()
        for done in ready { done() }
    }
}
