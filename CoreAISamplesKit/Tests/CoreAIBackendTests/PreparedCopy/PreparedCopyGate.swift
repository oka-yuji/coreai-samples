actor PreparedCopyGate {
    private let expected: Int
    private var arrivals = 0
    private var opened = false
    private var held: [CheckedContinuation<Void, Never>] = []
    private var watcher: CheckedContinuation<Void, Never>?

    init(expected: Int) {
        self.expected = expected
    }

    func arrive(waiting: Bool = true) async {
        arrivals += 1
        if arrivals == expected {
            watcher?.resume()
            watcher = nil
        }
        guard waiting, !opened else { return }
        await withCheckedContinuation { held.append($0) }
    }

    func allArrived() async {
        guard arrivals < expected else { return }
        await withCheckedContinuation { watcher = $0 }
    }

    func open() {
        opened = true
        for continuation in held {
            continuation.resume()
        }
        held = []
    }
}
