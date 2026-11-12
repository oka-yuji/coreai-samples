import FeatureStore
import Foundation

public enum ChronosJournalFeature: Feature {
    public struct PendingSave: Equatable, Sendable {
        public let ticket: Int
        public let batch: ChronosForecastBatch
    }

    public struct State: Equatable, Sendable {
        public var active = true
        public var ended = false
        public var presented = false
        public var entries: [ChronosJournalEntry] = []
        public var selectedID: UUID?
        public var detail: ChronosJournalDetail?
        public var ticket = 0
        public var busy = false
        public var checking = false
        public var error: ChronosDemoFailure?
        public var saveTicket = 0
        public var saves: [PendingSave] = []
        public var saving = false
        public var saveError: ChronosDemoFailure?
        public var savedID: UUID?
        public init() {}
    }

    public enum Action: Sendable {
        case capture(ChronosForecastBatch)
        case retrySave
        case saved(Int, UUID)
        case saveFailed(Int, ChronosDemoFailure)
        case appear, suspend, ended
        case open, close, back
        case select(UUID)
        case check
        case listed(Int, [ChronosJournalEntry])
        case loaded(Int, ChronosJournalDetail)
        case failed(Int, ChronosDemoFailure)
        case cancelled(Int)
    }

    public enum Effect: Equatable, Sendable {
        case save(Int, ChronosForecastBatch)
        case list(Int)
        case detail(Int, UUID)
        case check(Int, UUID)
    }

    public enum EffectID: Hashable, Sendable { case save, operation }

    public static func reduce(_ state: inout State, _ action: Action) -> [Command<Effect, EffectID>] {
        guard !state.ended else { return [] }
        switch action {
        case .capture(let batch):
            guard state.active, batch.mode == .live else { return [] }
            state.saveTicket += 1
            state.savedID = nil
            state.saves.append(PendingSave(ticket: state.saveTicket, batch: batch))
            return state.saving || state.saveError != nil ? [] : startSave(&state)
        case .retrySave:
            guard state.active, !state.saving else { return [] }
            return startSave(&state)
        case .saved(let ticket, let id):
            guard state.active, state.saving, state.saves.first?.ticket == ticket else { return [] }
            state.saves.removeFirst()
            state.savedID = id
            state.saving = false
            state.saveError = nil
            return startSave(&state)
        case .saveFailed(let ticket, let error):
            guard state.active, state.saving, state.saves.first?.ticket == ticket else { return [] }
            state.saving = false
            state.saveError = error
        case .appear:
            guard !state.active else { return [] }
            state.active = true
            let saves = state.saveError == nil ? startSave(&state) : []
            return saves + (state.presented ? refresh(&state) : [])
        case .suspend, .ended:
            state.active = false
            state.ended = {
                if case .ended = action { return true }
                return false
            }()
            state.ticket += 1
            state.busy = false
            state.checking = false
            state.saving = false
            return [.cancelAll]
        case .open:
            guard state.active else { return [] }
            state.presented = true
            return refresh(&state)
        case .close:
            state.presented = false
            state.ticket += 1
            state.busy = false
            state.checking = false
            return [.cancel(.operation)]
        case .back:
            guard state.active, state.presented else { return [] }
            state.selectedID = nil
            state.detail = nil
            return refresh(&state)
        case .select(let id):
            guard state.active, state.presented else { return [] }
            state.selectedID = id
            state.detail = nil
            return refresh(&state)
        case .check:
            guard state.active, state.presented, !state.busy, let id = state.selectedID,
                state.detail?.record.id == id
            else { return [] }
            state.ticket += 1
            state.busy = true
            state.checking = true
            state.error = nil
            return [.run(.check(state.ticket, id), id: .operation)]
        case .listed(let ticket, let entries):
            guard accepts(state, ticket), state.selectedID == nil else { return [] }
            state.entries = entries
            state.busy = false
        case .loaded(let ticket, let detail):
            guard accepts(state, ticket), state.selectedID == detail.record.id else { return [] }
            state.detail = detail
            state.busy = false
            state.checking = false
        case .failed(let ticket, let error):
            guard accepts(state, ticket) else { return [] }
            state.error = error
            state.busy = false
            state.checking = false
        case .cancelled(let ticket):
            guard accepts(state, ticket) else { return [] }
            state.busy = false
            state.checking = false
        }
        return []
    }

    private static func accepts(_ state: State, _ ticket: Int) -> Bool {
        state.active && state.presented && state.ticket == ticket
    }

    private static func startSave(_ state: inout State) -> [Command<Effect, EffectID>] {
        guard let pending = state.saves.first else { return [] }
        state.saving = true
        state.saveError = nil
        return [.run(.save(pending.ticket, pending.batch), id: .save)]
    }

    private static func refresh(_ state: inout State) -> [Command<Effect, EffectID>] {
        state.ticket += 1
        state.busy = true
        state.checking = false
        state.error = nil
        if let id = state.selectedID { return [.run(.detail(state.ticket, id), id: .operation)] }
        return [.run(.list(state.ticket), id: .operation)]
    }
}
