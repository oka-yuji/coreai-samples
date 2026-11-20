@MainActor
final class SpokenPlaybackStop {
    private var finished = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func finish() {
        finished = true
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume() }
    }

    func wait() async {
        if !finished { await withCheckedContinuation { waiters.append($0) } }
    }
}
