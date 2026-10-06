import Observation

/// Holds a feature's state, reduces actions one at a time, and performs the commands the reducer returns.
/// MainActor is stated explicitly, so the store means the same whether the target's default isolation is
/// MainActor or nonisolated.
///
/// Guarantees (the numbers match the test names in StoreTests):
/// - G1: Actions are reduced one at a time in arrival order. An action sent during processing is queued.
/// - G2: A run with an id cancels the effect running under the same id before it starts.
/// - G3: Results that a cancelled effect returns later are dropped.
/// - G4: When the store is released, every running effect is cancelled.
/// - G5: If the state after reduce equals the old state, it is not assigned (and nothing is notified).
/// - G6: Failures come back as actions too. This is a promise of the runner, not of the store (see `Run`).
/// - G7: An effect cancelled before it starts never runs.
/// - G8: `.cancelAll` cancels every running effect. Commands are handled in the order they are returned.
@MainActor
@Observable
final class Store<F: Feature> {
    /// Runs one effect and reports its results by calling `send` zero or more times. `send` is non-escaping,
    /// so it can be called only while the effect runs. The runner runs on the MainActor, so do not put heavy
    /// work directly here (convention A9). Use `@concurrent` for heavy computation that holds no state.
    /// Confine operations on stateful OS resources (such as a camera) to an actor or a dedicated serial queue.
    ///
    /// G6: The runner promises to return failures as actions. The type cannot enforce it (`catch {}` still
    /// compiles), so check it with a failure test for each effect.
    typealias Run = @MainActor (_ effect: F.Effect, _ send: (F.Action) -> Void) async -> Void

    private(set) var state: F.State

    @ObservationIgnored private let run: Run
    /// Effects in flight, keyed by token. An effect can deliver actions only while it is listed here.
    @ObservationIgnored private var running: [Int: (id: F.EffectID?, task: Task<Void, Never>)] = [:]
    @ObservationIgnored private var nextToken = 0
    @ObservationIgnored private var queue: [F.Action] = []
    @ObservationIgnored private var isProcessing = false

    init(initialState: F.State, run: @escaping Run) {
        state = initialState
        self.run = run
    }

    isolated deinit {
        cancelAll()  // G4
    }

    /// Reduces `action`, and any action sent while it is processed, one at a time in arrival order (G1).
    func send(_ action: F.Action) {
        queue.append(action)
        guard !isProcessing else { return }  // A reentrant send only queues; the outer loop handles it in turn.
        isProcessing = true
        defer { isProcessing = false }

        while !queue.isEmpty {
            var next = state
            let commands = F.reduce(&next, queue.removeFirst())
            if next != state { state = next }  // G5: skip the assignment (and the notification) when equal.
            commands.forEach(perform)  // G8: handle them in the order returned.
        }
    }

    private func perform(_ command: Command<F.Effect, F.EffectID>) {
        switch command {
        case .run(let effect, let id):
            if let id { cancel(id) }  // G2
            start(effect, id: id)
        case .cancel(let id):
            cancel(id)
        case .cancelAll:
            cancelAll()
        }
    }

    private func start(_ effect: F.Effect, id: F.EffectID?) {
        let token = nextToken
        nextToken += 1
        let run = run
        // State @MainActor explicitly. Then tasks start on the MainActor in the order they were created (SE-0431).
        // With implicit isolation, the body could start off the MainActor first and slip past the G7 check below.
        // self is weak so that an effect in flight does not keep the store alive (G4).
        let task = Task { @MainActor [weak self] in
            defer { self?.running[token] = nil }
            // G7: do not run an effect cancelled (or whose store was released) before it started.
            guard !Task.isCancelled, self != nil else { return }
            await run(effect) { self?.deliver($0, from: token) }
        }
        running[token] = (id, task)
    }

    private func cancel(_ id: F.EffectID) {
        // ponytail: linear scan; a few effects in flight are expected. Keep an id-to-token index if that grows.
        for (token, effect) in running where effect.id == id {
            effect.task.cancel()
            running[token] = nil
        }
    }

    /// Cancels every effect in flight and empties the ledger. Results they return afterwards are dropped (G3).
    private func cancelAll() {
        for effect in running.values { effect.task.cancel() }
        running.removeAll()
    }

    private func deliver(_ action: F.Action, from token: Int) {
        guard running[token] != nil else { return }  // G3: drop results from a cancelled or replaced effect.
        send(action)
    }
}
