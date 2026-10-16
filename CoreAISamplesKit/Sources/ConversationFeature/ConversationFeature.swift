import FeatureStore
import Foundation

public enum ConversationAction: Equatable, Sendable {
    case library(ConversationLibraryAction)
    case modeChanged(ConversationMode)
    case settingsChanged(ConversationSettings)
    case send(String, mode: ConversationMode, now: Date)
    case sendRepair(String, now: Date)
    case transcriptChanged(run: Int, TranscriptRecord)
    case toolRequested(run: Int, ToolCallRecord)
    case guardNote(run: Int, title: String, text: String, warning: Bool)
    case stepFinished(run: Int, StepStats)
    case generationEnded(run: Int, stopReason: String?)
    case generationFailed(run: Int, message: String)
    case evidenceChanged(run: Int, ConversationEvidence)
    case preparationChanged(run: Int, processed: Int, total: Int)
    case modelPrepared(run: Int, name: String, location: String)
    case stopRequested
    case restoreLatestRequested
    case latestRestored(ArchivedConversation?)
    case archiveSaved(id: String, revision: Int)
    case archiveFailed(id: String, revision: Int, message: String)
    case retrySave
    case saveCopyRequested(id: String, now: Date)
    case terminationRequested
    case terminationCancelled
    case stopped(run: Int)
    case restoreRequested(id: String)
    case restored(ConversationSnapshot)
    case restoreFailed(id: String?, message: String)
    case archiveRestored(id: String, ArchivedConversation)
    case saved(id: String)
    case failed(id: String, message: String)
}

public enum ConversationEffect: Equatable, Sendable {
    case library(ConversationLibraryEffect)
    case generate(run: Int, prompt: String, policy: TurnPolicy, finalize: Bool)
    case runTool(ToolCallRecord, finalize: Bool)
    case refuseTool(ToolCallRecord, reason: String)
    case stop(run: Int)
    case save(ConversationSnapshot)
    case restore(id: String)
    case restoreLatest
    case terminationReady(Bool)
    case cancelRepair
}

public enum ConversationFeature: Feature {
    public enum EffectID: Hashable, Sendable {
        case generation, stop, save, restore, libraryList, libraryMutation
    }

    public typealias Command = FeatureStore.Command<ConversationEffect, EffectID>

    public static let finalizationNote = "remaining tool budget: 0 — answer now with what you have"
    public static let endedRefusal = "error: the turn ended before this tool ran"
    public static let contextFullPrefix = "Context window is full"
    public static let endedWithoutStopReason = "Generation ended without a stop reason"
    public static let stopReasonPrefix = "Generation ended with stop reason: "

    public static func contextFullMessage(needed: Int, limit: Int) -> String {
        "\(contextFullPrefix): the conversation needs \(needed) tokens and the limit is \(limit). "
            + "Start a new conversation, lower the page budget or ask for something narrower."
    }

    public static func contextFullNumbers(_ message: String) -> (needed: Int, limit: Int)? {
        let words = message.split(separator: " ")
        guard message.hasPrefix(contextFullPrefix), let needs = words.firstIndex(of: "needs"),
            let limit = words.lastIndex(of: "is"), needs + 1 < words.count, limit + 1 < words.count,
            let needed = Int(words[needs + 1]),
            let value = Int(words[limit + 1].trimmingCharacters(in: .punctuationCharacters))
        else { return nil }
        return (needed, value)
    }

    public static func finalized(_ text: String) -> String {
        text + "\n\n" + finalizationNote
    }

    public static func reduce(_ state: inout ConversationState, _ action: ConversationAction) -> [Command] {
        let commands = apply(&state, action)
        state.entries = ConversationEntry.render(state.transcript, state.turns, live: state.isRunning)
        return commands
    }

    public static func continueEligibility(_ state: ConversationState) -> Bool {
        guard !state.isRunning, let turn = state.turns.last, turn.finish == .cap else { return false }
        let entries = state.transcript.entries.dropFirst(turn.start + 1)
        guard let last = entries.last, !last.isReasoning,
            let answer = entries.compactMap(\.response).last?.trimmingCharacters(in: .whitespacesAndNewlines),
            !answer.isEmpty
        else { return false }
        guard let open = answer.range(of: "<tool_call>", options: .backwards) else { return true }
        return answer[open.upperBound...].contains("</tool_call>")
    }

    private static func apply(_ state: inout ConversationState, _ action: ConversationAction) -> [Command] {
        switch action {
        case .library(let action): return ConversationLibrary.reduce(&state, action)
        case .modeChanged(let mode):
            if !state.isRunning { state.mode = mode }
            return []
        case .settingsChanged(let settings):
            if !state.isRunning { state.settings = settings }
            return []
        case .send(let text, let mode, let now):
            return send(&state, text, mode: mode, now: now)
        case .sendRepair(let text, let now):
            return sendRepair(&state, text, now: now)
        case .transcriptChanged(let run, let transcript):
            guard state.isActive(run) else { return [] }
            state.transcript = transcript
            return countSteps(&state)
        case .toolRequested(let run, let call):
            return decide(&state, call, run: run)
        case .guardNote(let run, let title, let text, let warning):
            guard run == state.runID, !state.turns.isEmpty else { return [] }
            state.turns[state.turns.count - 1].notes.append(TurnNote(title: title, text: text, warning: warning))
            return save(&state)
        case .stepFinished(let run, let stats):
            guard state.isActive(run) else { return [] }
            state.turns[state.turns.count - 1].steps.append(stats)
            return save(&state)
        case .generationEnded(let run, let stopReason):
            guard state.isActive(run) else { return [] }
            let finish = Finish(stopReason: stopReason)
            let last = state.turns.count - 1
            if finish == .failed, state.turns[last].failure == nil {
                state.turns[last].failure =
                    stopReason.map { stopReasonPrefix + $0 } ?? endedWithoutStopReason
            }
            return end(&state, finish)
        case .generationFailed(let run, let message):
            guard state.isActive(run) else { return [] }
            state.turns[state.turns.count - 1].failure = message
            return end(&state, .failed)
        case .evidenceChanged(let run, let evidence):
            guard state.isActive(run) else { return [] }
            state.evidence = evidence
            return []
        case .modelPrepared(let run, let name, let location):
            guard state.isActive(run) else { return [] }
            state.modelID = name
            state.modelLocation = location
            return []
        case .preparationChanged(let run, let processed, let total):
            guard state.isActive(run), !state.stopRequested, total > 0 else { return [] }
            state.preparation = processed < total ? ConversationPreparation(processed: processed, total: total) : nil
            return []
        case .stopRequested:
            guard state.isRunning, !state.stopRequested else { return [] }
            state.stopRequested = true
            return [.run(.stop(run: state.runID), id: .stop)]
        case .stopped(let run):
            guard state.isActive(run) else { return [] }
            return end(&state, .stopped)
        case .restoreRequested(let id):
            guard state.canLeave || (state.isRestoring && !state.isTerminating) else { return [] }
            if !state.isRestoring { state.restoreBase = state.persistence }
            state.persistence = .restoring(id: id)
            return [.run(.restore(id: id), id: .restore)]
        case .restoreFailed(let id, let message):
            guard
                (id == nil && state.persistence == .restoringLatest)
                    || id.map({ state.persistence == .restoring(id: $0) }) == true
            else { return [] }
            state.persistence = state.restoreBase ?? .unsaved
            state.restoreBase = nil
            state.library.error = message
            return finishRestore(&state)
        case .archiveRestored(let id, let archived):
            guard state.persistence == .restoring(id: id) else { return [] }
            state.restoreBase = nil
            switch archived {
            case .snapshot(let snapshot): state.load(snapshot)
            case .readOnly(let id, let markdown, let reason):
                state.id = id
                state.title = state.library.items.first(where: { $0.id == id })?.title ?? id
                state.turns = []
                state.transcript = TranscriptRecord()
                state.evidence = ConversationEvidence()
                state.modelLocation = nil
                state.modelID = ""
                state.archiveRevision = 0
                state.readOnlyMarkdown = markdown
                state.persistence = .failed(reason)
            }
            return finishRestore(&state)
        case .restored(let snapshot):
            guard !state.isRunning, state.persistence == .restoring(id: snapshot.id) else { return [] }
            state.load(snapshot)
            return finishRestore(&state)
        case .restoreLatestRequested:
            guard state.canLeave, state.turns.isEmpty else { return [] }
            state.restoreBase = state.persistence
            state.persistence = .restoringLatest
            return [.run(.restoreLatest, id: .restore)]
        case .latestRestored(let archived):
            guard state.persistence == .restoringLatest else { return [] }
            state.restoreBase = nil
            switch archived {
            case .snapshot(let snapshot): state.load(snapshot)
            case .readOnly(let id, let markdown, let reason):
                state.id = id
                state.readOnlyMarkdown = markdown
                state.persistence = .failed(reason)
            case nil: state.persistence = .unsaved
            }
            if state.isTerminating { return finishRestore(&state) }
            return ConversationLibrary.reduce(&state, .refresh)
        case .archiveSaved(let id, let revision):
            guard id == state.id, state.persistence == .saving(revision: revision) else { return [] }
            state.persistence = .saved
            state.library.loading = nil
            if state.library.exportRetryID == id {
                state.library.exportRetryID = nil
                state.library.error = nil
            }
            state.library.upsert(ConversationSummary(ConversationSnapshot(state)))
            return state.isTerminating && !state.isRunning ? [.run(.terminationReady(true), id: nil)] : []
        case .archiveFailed(let id, let revision, let message):
            guard id == state.id, state.persistence == .saving(revision: revision) else { return [] }
            state.persistence = .failed(message)
            return state.isTerminating && !state.isRunning ? [.run(.terminationReady(false), id: nil)] : []
        case .retrySave:
            guard !state.isRunning, !state.isRestoring, state.readOnlyMarkdown == nil, !state.turns.isEmpty else {
                return []
            }
            return save(&state)
        case .saveCopyRequested(let id, let now):
            guard case .failed = state.persistence, !state.isRunning, state.readOnlyMarkdown == nil,
                !state.turns.isEmpty, !id.isEmpty, id != state.id
            else { return [] }
            state.id = id
            state.createdAt = now
            state.updatedAt = now
            state.archiveRevision = 0
            return save(&state)
        case .terminationRequested:
            guard !state.isTerminating else { return [] }
            state.isTerminating = true
            if state.isRunning {
                guard !state.stopRequested else { return [] }
                state.stopRequested = true
                return [.run(.stop(run: state.runID), id: .stop)]
            }
            if state.isSaving || state.isRestoring || state.library.isMutating { return [] }
            if state.turns.isEmpty || state.readOnlyMarkdown != nil || state.persistence == .saved {
                return [.run(.terminationReady(true), id: nil)]
            }
            return save(&state)
        case .terminationCancelled:
            state.isTerminating = false
            return []
        case .saved(let id):
            if id == state.id, !state.isRestoring { state.persistence = .saved }
            return []
        case .failed(let id, let message):
            if (state.persistence == .restoring(id: id) || state.persistence == .restoringLatest)
                || (id == state.id && !state.isRestoring)
            {
                state.persistence = .failed(message)
            }
            return []
        }
    }

    private static func finishRestore(_ state: inout ConversationState) -> [Command] {
        guard state.isTerminating else { return [] }
        state.isTerminating = false
        return apply(&state, .terminationRequested)
    }

    private static func send(
        _ state: inout ConversationState, _ text: String, mode: ConversationMode, now: Date
    ) -> [Command] {
        let prompt = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !state.isRunning, !state.isRestoring, !state.isTerminating, !state.library.isMutating,
            state.readOnlyMarkdown == nil, !prompt.isEmpty
        else { return [] }
        if state.title.isEmpty {
            state.title = String(prompt.split(whereSeparator: \.isNewline).joined(separator: " ").prefix(40))
        }
        state.mode = mode
        state.refusedSave = false
        let turn = ConversationTurn(
            prompt: prompt, mode: mode, settings: state.settings, start: state.transcript.entries.count)
        return start(&state, turn, now: now)
    }

    private static func sendRepair(_ state: inout ConversationState, _ text: String, now: Date) -> [Command] {
        if state.isTerminating { return [.run(.cancelRepair, id: nil)] }
        let prompt = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !state.isRunning, !state.isRestoring, !state.isTerminating, !state.library.isMutating,
            state.readOnlyMarkdown == nil,
            !prompt.isEmpty, let last = state.turns.last, last.mode == .work
        else { return [] }
        var turn = ConversationTurn(
            prompt: prompt, mode: .work, settings: last.settings, start: state.transcript.entries.count)
        turn.kind = .repair
        turn.offset = state.step
        return start(&state, turn, now: now)
    }

    private static func start(_ state: inout ConversationState, _ turn: ConversationTurn, now: Date) -> [Command] {
        var turn = turn
        let policy = turn.policy
        let finalize = policy.maxSteps - turn.offset <= 1
        if finalize { turn.finalization = .lastStep }
        state.turns.append(turn)
        state.updatedAt = now
        state.runID += 1
        state.isRunning = true
        state.stopRequested = false
        state.step = turn.offset + 1
        return [
            .run(.generate(run: state.runID, prompt: turn.prompt, policy: policy, finalize: finalize), id: .generation)
        ]
    }

    private static func countSteps(_ state: inout ConversationState) -> [Command] {
        let last = state.turns.count - 1
        let entries = state.transcript.entries
        guard let prompt = entries.indices.dropFirst(state.turns[last].start).first(where: { entries[$0].isPrompt })
        else { return [] }
        state.turns[last].start = prompt
        state.step =
            state.turns[last].offset + state.transcript.steps(from: prompt + 1).prefix { $0.outputs != nil }.count + 1
        let limit = state.turns[last].policy.maxSteps
        if state.step > limit { return cutOff(&state) }
        if state.step == limit, state.turns[last].finalization == .normal {
            state.turns[last].finalization = .lastStep
        }
        return []
    }

    private static func decide(_ state: inout ConversationState, _ call: ToolCallRecord, run: Int) -> [Command] {
        guard state.isActive(run), let policy = state.turns.last?.policy else {
            return [.run(.refuseTool(call, reason: endedRefusal), id: nil)]
        }
        let refusal = policy.refusal(for: call.name)
        if policy.refusedTools[call.name] != nil { state.refusedSave = true }
        if state.stopRequested || state.step >= policy.maxSteps {
            return [.run(.refuseTool(call, reason: refusal ?? endedRefusal), id: nil)] + cutOff(&state)
        }
        let finalize = state.step == policy.maxSteps - 1
        guard let refusal else { return [.run(.runTool(call, finalize: finalize), id: nil)] }
        return [.run(.refuseTool(call, reason: finalize ? finalized(refusal) : refusal), id: nil)]
    }

    private static func cutOff(_ state: inout ConversationState) -> [Command] {
        guard !state.stopRequested else { return [] }
        state.turns[state.turns.count - 1].finalization = .cutOff
        state.stopRequested = true
        return [.run(.stop(run: state.runID), id: .stop)]
    }

    private static func end(_ state: inout ConversationState, _ finish: Finish) -> [Command] {
        state.turns[state.turns.count - 1].finish = finish
        state.isRunning = false
        state.stopRequested = false
        state.preparation = nil
        return save(&state)
    }

    private static func save(_ state: inout ConversationState) -> [Command] {
        state.archiveRevision += 1
        state.persistence = .saving(revision: state.archiveRevision)
        return [.run(.save(ConversationSnapshot(state)), id: .save)]
    }
}
