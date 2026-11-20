@MainActor
final class SpokenTopicLeaseProbe {
    private(set) var acquisitions = 0
    private(set) var releases = 0
    var isRegistered = true
    private var waiter: CheckedContinuation<Void, Never>?
    private var continuation: CheckedContinuation<Void, Never>?
    private let delayed: Bool

    init(delayed: Bool) { self.delayed = delayed }

    func acquire() async -> @MainActor @Sendable () async -> Void {
        acquisitions += 1
        if delayed {
            await withCheckedContinuation {
                continuation = $0
                waiter?.resume()
                waiter = nil
            }
        }
        return { self.releases += 1 }
    }

    func waitForRequest() async {
        if continuation == nil { await withCheckedContinuation { waiter = $0 } }
    }

    func resume() {
        continuation?.resume()
        continuation = nil
    }
}
