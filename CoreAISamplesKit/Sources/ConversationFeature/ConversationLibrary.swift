import FeatureStore
import Foundation

public struct ConversationSummary: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var title: String
    public var updatedAt: Date
    public var mode: ConversationMode?
    public var revision: Int?

    public init(id: String, title: String, updatedAt: Date, mode: ConversationMode?, revision: Int?) {
        self.id = id
        self.title = title
        self.updatedAt = updatedAt
        self.mode = mode
        self.revision = revision
    }

    public init(_ snapshot: ConversationSnapshot) {
        self.init(
            id: snapshot.id, title: snapshot.title, updatedAt: snapshot.updatedAt,
            mode: snapshot.turns.last?.mode, revision: snapshot.revision)
    }
}

public struct ConversationLibraryState: Equatable, Sendable {
    public var items: [ConversationSummary] = []
    public var ticket = 0
    public var loading: Int?
    public var mutation: Int?
    public var pendingDelete: String?
    public var error: String?
    public var exportRetryID: String?

    public init() {}

    public var isMutating: Bool { mutation != nil }

    public mutating func upsert(_ item: ConversationSummary) {
        items.removeAll { $0.id == item.id }
        items.append(item)
        items.sort { $0.updatedAt == $1.updatedAt ? $0.id < $1.id : $0.updatedAt > $1.updatedAt }
    }
}

public enum ConversationLibraryAction: Equatable, Sendable {
    case refresh
    case loaded(ticket: Int, [ConversationSummary])
    case renameRequested(id: String, title: String, now: Date)
    case deleteRequested(id: String)
    case cancelDelete
    case deleteConfirmed(replacementID: String, now: Date)
    case renamed(ticket: Int, ConversationSnapshot, warning: String?)
    case retryExport
    case exportRepaired(ticket: Int)
    case deleted(ticket: Int, id: String, replacementID: String, now: Date, warning: String?)
    case selectNew
    case selected(ticket: Int)
    case failed(ticket: Int, message: String)
}

public enum ConversationLibraryEffect: Equatable, Sendable {
    case list(ticket: Int)
    case rename(ticket: Int, id: String, title: String, expectedRevision: Int, now: Date)
    case trash(ticket: Int, id: String, expectedRevision: Int?, replacementID: String, now: Date)
    case selectNew(ticket: Int)
    case repairExport(ticket: Int, id: String)
}

public enum ConversationLibrary {
    public static func reduce(_ state: inout ConversationState, _ action: ConversationLibraryAction)
        -> [ConversationFeature.Command]
    {
        switch action {
        case .refresh:
            guard !state.library.isMutating, !state.isTerminating else { return [] }
            state.library.ticket += 1
            state.library.loading = state.library.ticket
            if state.library.exportRetryID == nil { state.library.error = nil }
            return [.run(.library(.list(ticket: state.library.ticket)), id: .libraryList)]
        case .loaded(let ticket, let items):
            guard state.library.loading == ticket else { return [] }
            state.library.loading = nil
            if !state.library.isMutating {
                state.library.items = items
                if let retry = state.library.exportRetryID, !items.contains(where: { $0.id == retry }) {
                    state.library.exportRetryID = nil
                    state.library.error = nil
                }
            }
            return []
        case .renameRequested(let id, let title, let now):
            let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
            guard state.canLeave, !title.isEmpty, let item = state.library.items.first(where: { $0.id == id }),
                let revision = item.revision
            else { return [] }
            let ticket = begin(&state)
            return [
                .run(
                    .library(
                        .rename(
                            ticket: ticket, id: id, title: title,
                            expectedRevision: id == state.id ? state.archiveRevision : revision, now: now)),
                    id: .libraryMutation)
            ]
        case .deleteRequested(let id):
            guard state.canLeave, state.library.items.contains(where: { $0.id == id }) else { return [] }
            state.library.pendingDelete = id
            return []
        case .cancelDelete:
            state.library.pendingDelete = nil
            return []
        case .deleteConfirmed(let replacementID, let now):
            guard state.canLeave, let id = state.library.pendingDelete,
                let item = state.library.items.first(where: { $0.id == id })
            else { return [] }
            state.library.pendingDelete = nil
            let ticket = begin(&state)
            return [
                .run(
                    .library(
                        .trash(
                            ticket: ticket, id: id,
                            expectedRevision: id == state.id && state.readOnlyMarkdown == nil
                                ? state.archiveRevision : item.revision,
                            replacementID: replacementID, now: now)), id: .libraryMutation)
            ]
        case .renamed(let ticket, let snapshot, let warning):
            guard state.library.mutation == ticket else { return [] }
            state.library.upsert(ConversationSummary(snapshot))
            state.library.error = warning
            if warning != nil {
                state.library.exportRetryID = snapshot.id
            } else if state.library.exportRetryID == snapshot.id {
                state.library.exportRetryID = nil
            }
            if state.id == snapshot.id {
                state.title = snapshot.title
                state.updatedAt = snapshot.updatedAt
                state.archiveRevision = snapshot.revision
                state.persistence = .saved
            }
            return finish(&state)
        case .deleted(let ticket, let id, let replacementID, let now, let warning):
            guard state.library.mutation == ticket else { return [] }
            state.library.items.removeAll { $0.id == id }
            if state.library.exportRetryID == id { state.library.exportRetryID = nil }
            state.library.error = warning
            if state.id == id {
                let library = state.library
                let terminating = state.isTerminating
                state = ConversationState(
                    id: replacementID, now: now, mode: state.mode, settings: state.settings,
                    modelID: state.modelID, instructionsVersion: state.instructionsVersion,
                    toolsVersion: state.toolsVersion)
                state.library = library
                state.isTerminating = terminating
            }
            return finish(&state)
        case .selectNew:
            guard state.canLeave, state.turns.isEmpty else { return [] }
            let ticket = begin(&state)
            return [.run(.library(.selectNew(ticket: ticket)), id: .libraryMutation)]
        case .retryExport:
            guard state.canLeave, let id = state.library.exportRetryID else { return [] }
            let ticket = begin(&state)
            return [.run(.library(.repairExport(ticket: ticket, id: id)), id: .libraryMutation)]
        case .exportRepaired(let ticket):
            guard state.library.mutation == ticket else { return [] }
            state.library.exportRetryID = nil
            state.library.error = nil
            return finish(&state)
        case .selected(let ticket):
            guard state.library.mutation == ticket else { return [] }
            return finish(&state)
        case .failed(let ticket, let message):
            if state.library.loading == ticket {
                state.library.loading = nil
                if !state.library.isMutating { state.library.error = message }
                return []
            }
            guard state.library.mutation == ticket else { return [] }
            state.library.error = message
            return finish(&state)
        }
    }

    private static func begin(_ state: inout ConversationState) -> Int {
        state.library.ticket += 1
        state.library.mutation = state.library.ticket
        state.library.loading = nil
        state.library.error = nil
        return state.library.ticket
    }

    private static func finish(_ state: inout ConversationState) -> [ConversationFeature.Command] {
        state.library.mutation = nil
        guard state.isTerminating else { return [] }
        state.isTerminating = false
        return ConversationFeature.reduce(&state, .terminationRequested)
    }
}

extension ConversationState {
    public var conversationSummaries: [ConversationSummary] {
        guard !turns.isEmpty || readOnlyMarkdown != nil, !library.items.contains(where: { $0.id == id }) else {
            return library.items
        }
        return [ConversationSummary(id: id, title: title, updatedAt: updatedAt, mode: lastMode, revision: nil)]
            + library.items
    }
}
