import Foundation

public enum ConversationMode: String, Codable, Sendable, CaseIterable {
    case chat
    case work
}

public struct ConversationSettings: Codable, Equatable, Sendable {
    public var effort: String
    public var maxSteps: Int
    public var maxTokens: Int?
    public var pageBudget: Int
    public var fileOperations: Bool
    public var citationGuard: Bool
    public var notesFolder: String

    public init(
        effort: String = "low", maxSteps: Int = 8, maxTokens: Int? = nil, pageBudget: Int = 10_000,
        fileOperations: Bool = false, citationGuard: Bool = true, notesFolder: String = ""
    ) {
        self.effort = effort
        self.maxSteps = maxSteps
        self.maxTokens = maxTokens
        self.pageBudget = pageBudget
        self.fileOperations = fileOperations
        self.citationGuard = citationGuard
        self.notesFolder = notesFolder
    }
}

public struct TurnPolicy: Equatable, Sendable {
    public static let tools: Set<String> = ["web_search", "fetch_page", "write_note", "move_note"]
    public static let savingRefusal = "error: saving notes needs Work mode"

    public var mode: ConversationMode
    public var settings: ConversationSettings
    public var allowedTools: Set<String>
    public var refusedTools: [String: String]
    public var maxSteps: Int
    public var maxTokens: Int
    public var citationGuard: Bool
    public var collapsesSteps: Bool

    public init(_ mode: ConversationMode, settings: ConversationSettings) {
        self.mode = mode
        self.settings = settings
        switch mode {
        case .chat:
            allowedTools = ["web_search", "fetch_page"]
            refusedTools = ["write_note": Self.savingRefusal, "move_note": Self.savingRefusal]
            maxSteps = 2
            maxTokens = settings.maxTokens ?? 2_048
            citationGuard = false
            collapsesSteps = true
        case .work:
            allowedTools = Self.tools
            refusedTools = [:]
            maxSteps = settings.maxSteps
            maxTokens = settings.maxTokens ?? 4_096
            citationGuard = settings.citationGuard
            collapsesSteps = false
        }
    }

    public func refusal(for tool: String) -> String? {
        if allowedTools.contains(tool) { return nil }
        return refusedTools[tool] ?? "error: \(tool) is not available in \(mode.rawValue) mode"
    }
}

public enum Finish: String, Codable, Sendable {
    case none, eos, cap, stopped, failed

    public init(stopReason: String?) {
        switch stopReason {
        case "stopToken": self = .eos
        case "maxTokens": self = .cap
        default: self = .failed
        }
    }
}

public enum Finalization: String, Codable, Sendable {
    case normal, lastStep, cutOff
}

public enum TurnKind: String, Codable, Sendable {
    case normal, repair
}

public struct TurnNote: Codable, Equatable, Sendable {
    public var title: String
    public var text: String
    public var warning: Bool
}

public enum Persistence: Equatable, Sendable {
    case unsaved
    case saved
    case saving(revision: Int)
    case restoring(id: String)
    case restoringLatest
    case failed(String)
}

public struct StepStats: Codable, Equatable, Sendable {
    public var step: Int
    public var ttftSeconds: Double?
    public var tokensPerSecond: Double?
    public var promptTokens: Int
    public var reusedTokens: Int
    public var generatedTokens: Int
    public var toolCalls: Int
    public var finish: String

    public init(
        step: Int, ttftSeconds: Double?, tokensPerSecond: Double?, promptTokens: Int, reusedTokens: Int,
        generatedTokens: Int, toolCalls: Int, finish: String
    ) {
        self.step = step
        self.ttftSeconds = ttftSeconds
        self.tokensPerSecond = tokensPerSecond
        self.promptTokens = promptTokens
        self.reusedTokens = reusedTokens
        self.generatedTokens = generatedTokens
        self.toolCalls = toolCalls
        self.finish = finish
    }

    public var line: String {
        [
            "step \(step)",
            ttftSeconds.map { String(format: "TTFT %.2fs", $0) } ?? "TTFT n/a",
            String(format: "%.1f tok/s", tokensPerSecond ?? 0),
            "prompt \(promptTokens)",
            "reused \(reusedTokens)",
            "\(generatedTokens) tok",
            toolCalls == 0 ? "answered" : "\(toolCalls) tool call(s)",
            finish,
        ].joined(separator: "  |  ")
    }
}

public struct ConversationTurn: Codable, Equatable, Sendable {
    public var prompt: String
    public var mode: ConversationMode
    public var settings: ConversationSettings
    public var start: Int
    public var steps: [StepStats] = []
    public var finish = Finish.none
    public var finalization = Finalization.normal
    public var failure: String?
    public var kind = TurnKind.normal
    public var offset = 0
    public var notes: [TurnNote] = []

    public var policy: TurnPolicy {
        TurnPolicy(mode, settings: settings)
    }
}

public struct ConversationState: Equatable, Sendable {
    public var id: String
    public var title = ""
    public var mode: ConversationMode
    public var settings: ConversationSettings
    public var modelID: String
    public var modelLocation: String?
    public var instructionsVersion: String
    public var toolsVersion: String
    public var createdAt: Date
    public var updatedAt: Date
    public var transcript = TranscriptRecord()
    public var evidence = ConversationEvidence()
    public var turns: [ConversationTurn] = []
    public var entries: [ConversationEntry] = []
    public var isRunning = false
    public var stopRequested = false
    public var runID = 0
    public var step = 0
    public var refusedSave = false
    public var persistence = Persistence.unsaved
    public var archiveRevision = 0
    public var preparation: ConversationPreparation?
    public var readOnlyMarkdown: String?
    public var isTerminating = false

    public init(
        id: String, now: Date, mode: ConversationMode = .chat, settings: ConversationSettings = ConversationSettings(),
        modelID: String = "", instructionsVersion: String = "", toolsVersion: String = ""
    ) {
        self.id = id
        self.mode = mode
        self.settings = settings
        self.modelID = modelID
        self.instructionsVersion = instructionsVersion
        self.toolsVersion = toolsVersion
        createdAt = now
        updatedAt = now
    }

    public var lastMode: ConversationMode? {
        turns.last?.mode
    }

    public var finalization: Finalization {
        turns.last?.finalization ?? Finalization.normal
    }

    public var lastFinish: Finish {
        turns.last?.finish ?? Finish.none
    }

    public var isRestoring: Bool {
        switch persistence {
        case .restoring, .restoringLatest: true
        default: false
        }
    }

    public var isSaving: Bool {
        if case .saving = persistence { true } else { false }
    }

    public var canLeave: Bool {
        guard !isRunning, !isSaving, !isRestoring, !isTerminating else { return false }
        if case .failed = persistence { return turns.isEmpty || readOnlyMarkdown != nil }
        return true
    }

    func isActive(_ run: Int) -> Bool {
        isRunning && run == runID
    }

    mutating func load(_ snapshot: ConversationSnapshot) {
        id = snapshot.id
        title = snapshot.title
        createdAt = snapshot.createdAt
        updatedAt = snapshot.updatedAt
        modelID = snapshot.modelID
        modelLocation = snapshot.modelLocation
        instructionsVersion = snapshot.instructionsVersion
        toolsVersion = snapshot.toolsVersion
        transcript = snapshot.transcript
        evidence = snapshot.evidence ?? ConversationEvidence()
        turns = snapshot.turns
        if let last = turns.indices.last, turns[last].finish == Finish.none { turns[last].finish = .stopped }
        mode = turns.last?.mode ?? mode
        settings = turns.last?.settings ?? settings
        step = 0
        refusedSave = false
        persistence = .saved
        archiveRevision = snapshot.revision
        preparation = nil
        readOnlyMarkdown = nil
    }
}

public enum ArchivedConversation: Equatable, Sendable {
    case snapshot(ConversationSnapshot)
    case readOnly(id: String, markdown: String, reason: String)
}

public struct ConversationPreparation: Equatable, Sendable {
    public var processed: Int
    public var total: Int

    public init(processed: Int, total: Int) {
        self.processed = processed
        self.total = total
    }
}

extension ConversationState {
    public var contextFull: Bool {
        turns.last?.failure?.hasPrefix(ConversationFeature.contextFullPrefix) == true
    }

    public var status: ConversationStatus? {
        guard let turn = turns.last else { return nil }
        let limit = turn.policy.maxSteps
        if isRunning {
            if stopRequested { return .stopping }
            if case .toolCalls(_, let calls, let metadata)? = transcript.entries.last, metadata != nil,
                transcript.entries.count > turn.start + 1
            {
                return .running(step: step, tools: calls.map(\.name))
            }
            return .generating(step: step, limit: limit)
        }
        if turn.finalization == .cutOff { return .cutOff(limit: limit) }
        if turn.finish == .stopped { return .stopped(steps: step) }
        if turn.failure != nil { return contextFull ? .overflow(step: step) : .failed(step: step) }
        return turn.steps.last.map(ConversationStatus.finished)
    }

    public var statusLine: String { status?.line ?? "" }

    public var contextEndReserved: Int? {
        guard !isRunning, let turn = turns.last, turn.finish == .cap else { return nil }
        let reserved = transcript.entries.reversed().lazy.compactMap { entry -> Int? in
            if case .response(_, _, let metadata) = entry { metadata?.reservedTokens } else { nil }
        }.first
        guard let reserved, reserved < turn.policy.maxTokens else { return nil }
        return reserved
    }
}

public enum ConversationStatus: Equatable, Sendable {
    case generating(step: Int, limit: Int)
    case running(step: Int, tools: [String])
    case stopping
    case cutOff(limit: Int)
    case stopped(steps: Int)
    case failed(step: Int)
    case overflow(step: Int)
    case finished(StepStats)

    public var line: String {
        switch self {
        case .generating(let step, let limit): "Step \(step) of \(limit): generating…"
        case .running(let step, let tools): "Step \(step): running " + tools.joined(separator: " + ") + "…"
        case .stopping: "Stopping…"
        case .cutOff(let limit): "Stopped at the \(limit)-step limit."
        case .stopped(let steps): "Stopped after \(steps) step(s)."
        case .failed(let step): "Failed at step \(step)."
        case .overflow(let step): "Context overflow at step \(step)."
        case .finished(let stats): stats.line
        }
    }
}
