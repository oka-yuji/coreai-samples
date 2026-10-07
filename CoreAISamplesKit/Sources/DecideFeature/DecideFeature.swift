import FeatureStore
import Foundation

public struct DecideState: Sendable, Equatable {
    public enum Phase: Sendable, Equatable {
        case idle, loading, running, stopping, reading, closed
    }

    public var input: DecideInput
    public var model: DecideBundleInfo?
    public var examples: [DecideExample] = []
    public var loadingExamples = false
    public var exampleSession = 0
    public var phase: Phase = .idle
    public var session = 0
    public var revision = 0
    public var presentationRevision = 0
    public var resultRevision: Int?
    public var output: DecideOutput?
    public var previous: DecideOutput?
    public var confirmations: [Int: String] = [:]
    public var changedItems: Set<Int> = []
    public var lowestFirst = false
    public var inputTokens: Int?
    public var countError: String?
    public var message: String?
    public var initialSeconds: Double?
    public var reuseSeconds: Double?
    public var exportData: Data?
    public var exportTicket: DecideExportTicket?
    public var nextExportID = 0
    public var exportDismissed = false
    public var nextItemID: Int

    public init(input: DecideInput = DecideInput(document: "", items: [])) {
        self.input = input
        let maximum = input.items.map(\.id).max() ?? -1
        self.nextItemID = maximum == Int.max ? 0 : maximum + 1
    }

    public var busy: Bool { phase != .idle && phase != .closed }
    public var canRun: Bool { phase == .idle && model != nil && input.problem == nil && countError == nil }
    public var currentResult: Bool { output != nil && resultRevision == revision }
    public var rows: [DecideRow] {
        let rows = output?.rows ?? []
        guard lowestFirst else { return rows }
        return rows.enumerated().sorted {
            if $0.element.maximumProbability == $1.element.maximumProbability { return $0.offset < $1.offset }
            return $0.element.maximumProbability < $1.element.maximumProbability
        }.map(\.element)
    }
}

public enum DecideAction: Sendable, Equatable {
    case appeared
    case examplesLoaded(Int, [DecideExample])
    case examplesFailed(Int, String)
    case ended
    case openBundle(String)
    case loaded(Int, DecideBundleInfo)
    case setDocument(String)
    case replaceInput(DecideInput)
    case addItem
    case updateItem(DecideItem)
    case removeItem(Int)
    case moveItem(Int, Int)
    case readDocument(URL)
    case documentRead(Int, String)
    case counted(Int, Int, Int)
    case countFailed(Int, Int, String)
    case run(reuse: Bool)
    case completed(Int, Int, DecideOutput)
    case failed(Int, String)
    case stop
    case stopped(Int)
    case setLowestFirst(Bool)
    case confirm(Int, String?)
    case prepareExport
    case exportPrepared(DecideExportTicket, Data)
    case exportFailed(DecideExportTicket, String)
    case dismissExport(DecideExportTicket)
    case exportFinished(DecideExportTicket, String?)
    case uiError(String)
}

public enum DecideEffect: Sendable, Equatable {
    case examples(Int)
    case load(Int, String)
    case count(Int, Int, DecideInput)
    case read(Int, URL)
    case decide(Int, Int, DecideInput, reuse: Bool)
    case waitForIdle(Int)
    case export(DecideExportTicket, DecideOutput, [Int: String])
}

public enum DecideEffectID: Sendable, Hashable {
    case operation, count, stop, export, examples
}

public enum DecideFeature: Feature {
    public typealias State = DecideState
    public typealias Action = DecideAction
    public typealias Effect = DecideEffect
    public typealias EffectID = DecideEffectID
    public typealias Command = FeatureStore.Command<Effect, EffectID>

    public static func reduce(_ state: inout State, _ action: Action) -> [Command] {
        if state.phase == .closed {
            guard case .appeared = action else { return [] }
        }
        switch action {
        case .appeared:
            if state.phase == .closed { state.phase = .idle }
            var commands = count(state)
            if state.examples.isEmpty && !state.loadingExamples {
                state.exampleSession += 1
                state.loadingExamples = true
                commands.append(.run(.examples(state.exampleSession), id: .examples))
            }
            return commands
        case .examplesLoaded(let session, let examples):
            guard state.exampleSession == session, state.loadingExamples else { return [] }
            state.loadingExamples = false
            state.examples = examples
            if state.input.document.isEmpty && state.input.items.isEmpty, let first = examples.first {
                state.input = first.input
                state.nextItemID = (first.input.items.map(\.id).max() ?? -1) + 1
                invalidate(&state, document: true)
                return count(state)
            }
        case .examplesFailed(let session, let message):
            guard state.exampleSession == session, state.loadingExamples else { return [] }
            state.loadingExamples = false
            state.message = message
        case .ended:
            state.session += 1
            state.phase = .closed
            state.loadingExamples = false
            state.exportData = nil
            state.exportTicket = nil
            return [.cancelAll]
        case .openBundle(let path):
            guard !state.busy else { return [] }
            state.session += 1
            state.phase = .loading
            state.model = nil
            state.output = nil
            state.previous = nil
            state.resultRevision = nil
            state.changedItems = []
            state.initialSeconds = nil
            state.reuseSeconds = nil
            state.exportData = nil
            state.exportTicket = nil
            state.confirmations = [:]
            state.message = nil
            state.inputTokens = nil
            state.countError = nil
            return [.cancel(.count), .run(.load(state.session, path), id: .operation)]
        case .loaded(let session, let info):
            guard state.session == session, state.phase == .loading else { return [] }
            state.model = info
            state.phase = .idle
            return count(state)
        case .setDocument(let text):
            guard !state.busy, !state.input.document.utf8.elementsEqual(text.utf8) else { return [] }
            state.input.document = text
            invalidate(&state, document: true)
            return count(state)
        case .replaceInput(let input):
            guard !state.busy else { return [] }
            state.input = input
            let maximum = input.items.map(\.id).max() ?? -1
            state.nextItemID = maximum == Int.max ? 0 : maximum + 1
            invalidate(&state, document: true)
            return count(state)
        case .addItem:
            guard !state.busy, state.input.items.count < 16, state.nextItemID < Int.max else { return [] }
            state.input.items.append(.blank(id: state.nextItemID))
            state.nextItemID += 1
            invalidate(&state)
            return count(state)
        case .updateItem(let item):
            guard !state.busy, let index = state.input.items.firstIndex(where: { $0.id == item.id }) else { return [] }
            guard state.input.items[index] != item else { return [] }
            state.input.items[index] = item
            invalidate(&state)
            return count(state)
        case .removeItem(let id):
            guard !state.busy, let index = state.input.items.firstIndex(where: { $0.id == id }) else { return [] }
            state.input.items.remove(at: index)
            invalidate(&state)
            return count(state)
        case .moveItem(let id, let delta):
            guard !state.busy, let from = state.input.items.firstIndex(where: { $0.id == id }),
                state.input.items.indices.contains(from + delta)
            else { return [] }
            let item = state.input.items.remove(at: from)
            state.input.items.insert(item, at: from + delta)
            invalidate(&state)
            return count(state)
        case .readDocument(let url):
            guard !state.busy else { return [] }
            state.session += 1
            state.phase = .reading
            state.message = nil
            return [.cancel(.count), .run(.read(state.session, url), id: .operation)]
        case .documentRead(let session, let text):
            guard state.session == session, state.phase == .reading else { return [] }
            state.phase = .idle
            state.input.document = text
            invalidate(&state, document: true)
            return count(state)
        case .counted(let session, let revision, let tokens):
            guard state.session == session, state.revision == revision, state.phase == .idle else { return [] }
            state.inputTokens = tokens
            state.countError = nil
        case .countFailed(let session, let revision, let message):
            guard state.session == session, state.revision == revision, state.phase == .idle else { return [] }
            state.inputTokens = nil
            state.countError = message
        case .run(let reuse):
            guard state.canRun else { return [] }
            state.session += 1
            state.phase = .running
            state.message = nil
            state.exportData = nil
            state.exportTicket = nil
            return [
                .cancel(.count),
                .run(.decide(state.session, state.revision, state.input, reuse: reuse), id: .operation),
            ]
        case .completed(let session, let revision, let output):
            guard state.session == session, state.revision == revision, state.phase == .running else { return [] }
            state.previous = state.output
            let before = Dictionary(uniqueKeysWithValues: (state.output?.rows ?? []).map { ($0.id, $0) })
            state.changedItems = Set(
                output.rows.filter { row in
                    guard let previous = before[row.id] else { return state.output != nil }
                    return !row.candidate.utf8.elementsEqual(previous.candidate.utf8)
                        || row.reviewFirst != previous.reviewFirst
                }.map(\.id))
            state.output = output
            state.resultRevision = revision
            state.phase = .idle
            state.inputTokens = output.inputTokens
            state.confirmations = [:]
            state.presentationRevision += 1
            if output.reusedDocument {
                state.reuseSeconds = output.seconds
            } else {
                state.initialSeconds = output.seconds + output.contextSeconds
            }
        case .failed(let session, let message):
            guard state.session == session,
                state.phase == .loading || state.phase == .running || state.phase == .reading
            else { return [] }
            state.phase = .idle
            state.message = message
        case .stop:
            guard state.busy, state.phase != .stopping else { return [] }
            state.session += 1
            state.phase = .stopping
            return [.cancel(.operation), .cancel(.count), .run(.waitForIdle(state.session), id: .stop)]
        case .stopped(let session):
            guard state.session == session, state.phase == .stopping else { return [] }
            state.phase = .idle
            state.message = "Stopped."
        case .setLowestFirst(let value):
            state.lowestFirst = value
        case .confirm(let id, let option):
            guard state.currentResult, !state.busy, let row = state.output?.rows.first(where: { $0.id == id }) else {
                return []
            }
            if let option, !row.probabilities.contains(where: { $0.option.utf8.elementsEqual(option.utf8) }) {
                return []
            }
            state.confirmations[id] = option
            state.presentationRevision += 1
            state.exportTicket = nil
            state.exportData = nil
        case .prepareExport:
            guard state.currentResult, !state.busy, let output = state.output else { return [] }
            state.message = nil
            state.exportData = nil
            state.nextExportID += 1
            state.exportDismissed = false
            let ticket = DecideExportTicket(
                id: state.nextExportID, session: state.session, revision: state.revision,
                presentation: state.presentationRevision)
            state.exportTicket = ticket
            return [.run(.export(ticket, output, state.confirmations), id: .export)]
        case .exportPrepared(let ticket, let data):
            guard currentExport(ticket, state), !state.exportDismissed, state.currentResult, !state.busy
            else { return [] }
            state.exportData = data
        case .exportFailed(let ticket, let message):
            guard currentExport(ticket, state), !state.exportDismissed, state.exportData == nil else { return [] }
            state.exportData = nil
            state.exportTicket = nil
            state.message = message
        case .dismissExport(let ticket):
            guard currentExport(ticket, state) else { return [] }
            state.exportData = nil
            state.exportDismissed = true
        case .uiError(let message):
            if !state.busy { state.message = message }
        case .exportFinished(let ticket, let error):
            guard currentExport(ticket, state) else { return [] }
            state.exportData = nil
            state.exportTicket = nil
            state.message = error ?? "JSON exported."
        }
        return []
    }

    private static func currentExport(_ ticket: DecideExportTicket, _ state: State) -> Bool {
        ticket == state.exportTicket && ticket.session == state.session && ticket.revision == state.revision
            && ticket.presentation == state.presentationRevision
    }

    private static func count(_ state: State) -> [Command] {
        guard state.phase == .idle, state.model != nil, state.input.problem == nil else { return [.cancel(.count)] }
        return [.run(.count(state.session, state.revision, state.input), id: .count)]
    }

    private static func invalidate(_ state: inout State, document: Bool = false) {
        state.revision += 1
        state.resultRevision = nil
        state.confirmations = [:]
        state.inputTokens = nil
        state.countError = nil
        state.message = nil
        state.exportData = nil
        state.exportTicket = nil
        if document {
            state.output = nil
            state.previous = nil
            state.changedItems = []
            state.initialSeconds = nil
            state.reuseSeconds = nil
        }
    }
}
