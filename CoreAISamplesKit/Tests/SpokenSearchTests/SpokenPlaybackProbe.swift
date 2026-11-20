import SpeechRuntime

@MainActor
final class SpokenPlaybackProbe: PlaybackSession {
    let interruptsOnStart: Bool
    private(set) var isActive = false
    private(set) var starts = 0
    private var waiters: [@MainActor () -> Void] = []
    private var observed: CheckedContinuation<Void, Never>?

    init(interruptsOnStart: Bool = false) { self.interruptsOnStart = interruptsOnStart }

    func start(
        interrupted: @escaping @MainActor () -> Void,
        ready: @escaping @MainActor (Result<Void, any Error>) -> Void
    ) {
        starts += 1
        isActive = true
        ready(.success(()))
        if interruptsOnStart { interrupted() }
    }

    func stop(completed: @escaping @MainActor () -> Void) {
        guard isActive else {
            completed()
            return
        }
        if interruptsOnStart {
            isActive = false
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
