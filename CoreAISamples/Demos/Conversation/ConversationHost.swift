import ConversationFeature
import Foundation
import Observation

@MainActor
@Observable
final class ConversationPreferences {
    static let effortKey = "agent.reasoningEffort"
    static let maxStepsKey = "agent.maxSteps"
    static let maxTokensKey = "agent.maxTokens"
    static let pageBudgetKey = "agent.pageBudget"
    static let braveKeyKey = "agent.braveAPIKey"
    static let fileOpsKey = "agent.allowFileOps"
    static let notesFolderKey = "agent.notesFolder"
    static let citationGuardKey = "agent.citationGuard"
    static let citeFetchedKey = "agent.citeFetched"
    static let maxTokenChoices = [512, 1_024, 1_536, 2_048, 4_096, 8_192]

    var persists: Bool
    var effort: AgentReasoningEffort { didSet { store(effort.rawValue, Self.effortKey) } }
    var maxSteps: Int { didSet { store(maxSteps, Self.maxStepsKey) } }
    var maxTokens: Int? { didSet { store(maxTokens, Self.maxTokensKey) } }
    var pageBudget: Int { didSet { store(pageBudget, Self.pageBudgetKey) } }
    var braveAPIKey: String { didSet { store(braveAPIKey, Self.braveKeyKey) } }
    var fileOperations: Bool { didSet { store(fileOperations, Self.fileOpsKey) } }
    var notesFolderPath: String { didSet { store(notesFolderPath, Self.notesFolderKey) } }
    var citationGuard: Bool { didSet { store(citationGuard, Self.citationGuardKey) } }

    @ObservationIgnored var citeFetched: Bool { didSet { store(citeFetched, Self.citeFetchedKey) } }
    @ObservationIgnored var sourcesRequired: Bool?
    @ObservationIgnored var stableNotePaths = false
    @ObservationIgnored var fixedDate: String?
    @ObservationIgnored var cassette: ToolCassette?
    @ObservationIgnored var deadline: Date?
    @ObservationIgnored private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard, persists: Bool = true) {
        self.defaults = defaults
        self.persists = persists
        func positive(_ key: String) -> Int? { (defaults.object(forKey: key) as? Int).flatMap { $0 > 0 ? $0 : nil } }
        effort = defaults.string(forKey: Self.effortKey).flatMap(AgentReasoningEffort.init(rawValue:)) ?? .low
        maxSteps = positive(Self.maxStepsKey) ?? 8
        maxTokens = positive(Self.maxTokensKey)
        pageBudget = positive(Self.pageBudgetKey) ?? 10_000
        braveAPIKey = defaults.string(forKey: Self.braveKeyKey) ?? ""
        fileOperations = defaults.bool(forKey: Self.fileOpsKey)
        notesFolderPath = defaults.string(forKey: Self.notesFolderKey) ?? ""
        citationGuard = defaults.object(forKey: Self.citationGuardKey) as? Bool ?? true
        citeFetched = defaults.object(forKey: Self.citeFetchedKey) as? Bool ?? true
    }

    private func store(_ value: Any?, _ key: String) {
        guard persists else { return }
        if let value { defaults.set(value, forKey: key) } else { defaults.removeObject(forKey: key) }
    }

    var settings: ConversationSettings {
        ConversationSettings(
            effort: effort.rawValue, maxSteps: maxSteps, maxTokens: maxTokens, pageBudget: pageBudget,
            fileOperations: fileOperations, citationGuard: citationGuard, notesFolder: notesFolderPath)
    }

    var searchProvider: String { braveAPIKey.isEmpty ? "DuckDuckGo" : "Brave Search API" }

    var notesDirectory: URL {
        notesFolderPath.isEmpty
            ? AgentTools.defaultNotesDirectory()
            : URL(fileURLWithPath: (notesFolderPath as NSString).expandingTildeInPath, isDirectory: true)
    }

    var toolSettings: AgentToolSettings {
        AgentToolSettings(
            pageCharacterBudget: pageBudget, braveAPIKey: braveAPIKey, notesDirectory: notesDirectory,
            allowFileOperations: fileOperations, cassette: cassette, stableNotePaths: stableNotePaths)
    }
}

@MainActor
@Observable
final class ConversationHost {
    static let continuePrompt = "Continue the previous request in Work mode. Reuse what you already fetched."

    let chat: ChatViewModel
    let preferences: ConversationPreferences
    private(set) var store: ConversationStore
    @ObservationIgnored private(set) var runner: ConversationRunner

    init(chat: ChatViewModel, preferences: ConversationPreferences = ConversationPreferences()) {
        self.chat = chat
        self.preferences = preferences
        (store, runner) = Self.make(chat: chat, preferences: preferences, mode: .chat)
    }

    var state: ConversationState { store.state }

    var contextFull: Bool { chat.contextFull || store.state.contextFull }

    var canSend: Bool {
        chat.phase == .ready && !store.state.isRunning && !chat.isResponding && !chat.externallyBusy && !contextFull
            && !chat.trimmedInput.isEmpty
    }

    func startNew() {
        guard !store.state.isRunning, !chat.isResponding else { return }
        chat.resetRAG()
        (store, runner) = Self.make(chat: chat, preferences: preferences, mode: store.state.mode)
    }

    @discardableResult
    func submit() -> Bool {
        guard canSend else { return false }
        if chat.ragIsOn {
            chat.reasoning = preferences.effort
            chat.maxTokens = TurnPolicy(.chat, settings: preferences.settings).maxTokens
            return chat.send(anchor: store.state.entries.count)
        }
        let text = chat.trimmedInput
        chat.input = ""
        return send(text, mode: store.state.mode)
    }

    @discardableResult
    func send(_ text: String, mode: ConversationMode) -> Bool {
        store.send(.settingsChanged(preferences.settings))
        store.send(.send(text, mode: mode, now: Date()))
        return store.state.isRunning
    }

    func continueInWork() {
        guard store.state.refusedSave, !store.state.isRunning, !contextFull else { return }
        send(Self.continuePrompt, mode: .work)
    }

    func stop() {
        if chat.isResponding { chat.stop() } else { store.send(.stopRequested) }
    }

    private static func make(chat: ChatViewModel, preferences: ConversationPreferences, mode: ConversationMode)
        -> (ConversationStore, ConversationRunner)
    {
        let now = Date()
        let id = newID(now)
        let runner = ConversationRunner(id: id, chat: chat, preferences: preferences)
        let state = ConversationState(
            id: id, now: now, mode: mode, settings: preferences.settings, modelID: chat.modelName,
            instructionsVersion: preferences.citeFetched ? "cite-fetched" : "frozen",
            toolsVersion: (AgentTools.names.dropLast(preferences.fileOperations ? 0 : 1)).joined(separator: ","))
        return (ConversationStore(initialState: state, run: runner.run), runner)
    }

    static func newID(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: date)
    }
}
