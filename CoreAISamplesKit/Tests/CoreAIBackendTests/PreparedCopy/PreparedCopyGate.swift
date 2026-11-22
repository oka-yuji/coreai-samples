actor PreparedCopyGate {
    private let expected: Int
    private var arrivals = 0
    private var opened = false
    private var held: [AsyncStream<Void>.Continuation] = []
    private var watcher: AsyncStream<Void>.Continuation?
    private(set) var entered = 0

    init(expected: Int) {
        self.expected = expected
    }

    func arrive(waiting: Bool = true) async throws {
        arrivals += 1
        if arrivals == expected {
            watcher?.finish()
            watcher = nil
        }
        guard waiting else { return }
        entered += 1
        guard !opened else { return }
        let (signal, continuation) = AsyncStream<Void>.makeStream()
        held.append(continuation)
        for await _ in signal {}
        try Task.checkCancellation()
    }

    func allArrived() async throws {
        guard arrivals < expected else { return }
        let (signal, continuation) = AsyncStream<Void>.makeStream()
        watcher = continuation
        for await _ in signal {}
        try Task.checkCancellation()
    }

    func open() {
        opened = true
        for continuation in held {
            continuation.finish()
        }
        held = []
    }
}
