import FeatureStore

public enum SearchKeyFeature: Feature {
    public struct State: Equatable, Sendable {
        public var saved = ""
        public var draft = ""
        public var phase = Phase.unloaded
        public var failure: Failure?
        public var ticket = 0

        public init() {}

        public var canEdit: Bool { phase == .ready }

        public var canSave: Bool { phase == .ready && (draft != saved || failure?.operation == .move) }

        public var status: Status {
            switch phase {
            case .unloaded, .loading: return .reading
            case .saving: return .saving
            case .ready: break
            }
            if let failure { return .failed(failure) }
            if draft != saved { return .unsaved }
            return saved.isEmpty ? .empty : .saved
        }
    }

    public enum Phase: Equatable, Sendable {
        case unloaded, loading, ready, saving
    }

    public enum Status: Equatable, Sendable {
        case reading, saving, unsaved, saved, empty
        case failed(Failure)
    }

    public struct Failure: Error, Equatable, Sendable {
        public enum Operation: Equatable, Sendable {
            case read, move, write, delete
        }

        public var operation: Operation
        public var reason: String

        public init(_ operation: Operation, reason: String) {
            self.operation = operation
            self.reason = reason
        }
    }

    public enum Load: Equatable, Sendable {
        case stored(String)
        case unmoved(String, Failure)
        case unreadable(Failure)
    }

    public enum Action: Equatable, Sendable {
        case loadRequested
        case loaded(ticket: Int, Load)
        case draftChanged(String)
        case saveRequested
        case saved(ticket: Int, Result<String, Failure>)
        case replaced(String)
    }

    public enum Effect: Equatable, Sendable {
        case load(ticket: Int)
        case save(String, ticket: Int)
    }

    public enum EffectID: Hashable, Sendable {
        case keychain
    }

    public static func reduce(_ state: inout State, _ action: Action) -> [Command<Effect, EffectID>] {
        switch action {
        case .loadRequested:
            guard state.phase == .unloaded || state.phase == .ready && state.failure?.operation == .read else {
                return []
            }
            state.phase = .loading
            state.ticket += 1
            return [.run(.load(ticket: state.ticket), id: .keychain)]
        case .loaded(let ticket, let load):
            guard ticket == state.ticket, state.phase == .loading else { return [] }
            switch load {
            case .stored(let key): (state.saved, state.failure) = (key, nil)
            case .unmoved(let key, let failure): (state.saved, state.failure) = (key, failure)
            case .unreadable(let failure): (state.saved, state.failure) = ("", failure)
            }
            state.draft = state.saved
            state.phase = .ready
            return []
        case .draftChanged(let text):
            state.draft = text
            return []
        case .saveRequested:
            guard state.canSave else { return [] }
            state.phase = .saving
            state.ticket += 1
            return [.run(.save(state.draft, ticket: state.ticket), id: .keychain)]
        case .saved(let ticket, let result):
            guard ticket == state.ticket, state.phase == .saving else { return [] }
            switch result {
            case .success(let key): (state.saved, state.failure) = (key, nil)
            case .failure(let failure): state.failure = failure
            }
            state.phase = .ready
            return []
        case .replaced(let key):
            (state.saved, state.draft, state.failure, state.phase) = (key, key, nil, .ready)
            state.ticket += 1
            return []
        }
    }
}
