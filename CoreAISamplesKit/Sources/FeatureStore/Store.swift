import Observation

@MainActor
@Observable
public final class Store<F: Feature> {

    public typealias Run = @MainActor (_ effect: F.Effect, _ send: (F.Action) -> Void) async -> Void

    public private(set) var state: F.State

    @ObservationIgnored private let run: Run

    @ObservationIgnored private var running: [Int: (id: F.EffectID?, task: Task<Void, Never>)] = [:]
    @ObservationIgnored private var nextToken = 0
    @ObservationIgnored private var queue: [F.Action] = []
    @ObservationIgnored private var isProcessing = false

    public init(initialState: F.State, run: @escaping Run) {
        state = initialState
        self.run = run
    }

    isolated deinit {
        cancelAll()
    }

    public func send(_ action: F.Action) {
        queue.append(action)
        guard !isProcessing else { return }
        isProcessing = true
        defer { isProcessing = false }

        while !queue.isEmpty {
            var next = state
            let commands = F.reduce(&next, queue.removeFirst())
            if next != state { state = next }
            commands.forEach(perform)
        }
    }

    private func perform(_ command: Command<F.Effect, F.EffectID>) {
        switch command {
        case .run(let effect, let id):
            if let id { cancel(id) }
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

        let task = Task { @MainActor [weak self] in
            defer { self?.running[token] = nil }

            guard !Task.isCancelled, self != nil else { return }
            await run(effect) { self?.deliver($0, from: token) }
        }
        running[token] = (id, task)
    }

    private func cancel(_ id: F.EffectID) {

        for (token, effect) in running where effect.id == id {
            effect.task.cancel()
            running[token] = nil
        }
    }

    private func cancelAll() {
        for effect in running.values { effect.task.cancel() }
        running.removeAll()
    }

    private func deliver(_ action: F.Action, from token: Int) {
        guard running[token] != nil else { return }
        send(action)
    }
}
