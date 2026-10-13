import ConversationFeature
import CoreAIBackend
import Foundation
import FoundationModels

struct TurnEnd {
    enum Kind {
        case ended(stopReason: String?)
        case failed(String)
        case stopped
    }

    static let answered = "answered"

    var kind: Kind
    var outcome: String
    var answer: String

    func action(_ run: Int) -> ConversationAction {
        switch kind {
        case .ended(let stopReason): .generationEnded(run: run, stopReason: stopReason)
        case .failed(let message): .generationFailed(run: run, message: message)
        case .stopped: .stopped(run: run)
        }
    }
}

@MainActor
final class ConversationSession {
    private enum Event: Sendable {
        case actions([ConversationAction])
        case stop
        case ended((any Error)?)
    }

    let id: String
    private(set) var modelName = ""
    private(set) var mode = ConversationMode.chat
    private(set) var effort = AgentReasoningEffort.low.rawValue
    private(set) var turnIndex = 0
    private(set) var divergeCount = 0
    private(set) var capSteps = 0
    private(set) var lastFinish: String?
    private(set) var lastGeneration: GenerationStats?
    private(set) var stopRequested = false
    private(set) var currentDate: String?

    private let chat: ChatViewModel
    private let preferences: ConversationPreferences
    private let tools: ToolRunner
    private var session: LanguageModelSession?
    private var model: LocalLanguageModel?
    private var sink: AsyncStream<Event>.Continuation?
    private var run = 0
    private var base = 0
    private var offset = 0
    private var completed = 0
    private var maxSteps = 0
    private var limitReached = false
    private var claimed: Set<String> = []
    private var turnMark: LanguageModelSession.Usage?
    private var stepMark: LanguageModelSession.Usage?
    private var started = ContinuousClock.now
    private var stepStart = ContinuousClock.now
    private var ttft: Double?
    private var firstEntryAt: ContinuousClock.Instant?
    private var previous: (input: Int, output: Int)?
    private var previousTokens: [Int32]?
    private var currentTokens: [Int32]?

    init(id: String, chat: ChatViewModel, preferences: ConversationPreferences, tools: ToolRunner) {
        self.id = id
        self.chat = chat
        self.preferences = preferences
        self.tools = tools
    }

    var step: Int { offset + completed + 1 }
    var usage: LanguageModelSession.Usage? { session?.usage }

    func generate(
        run: Int, prompt: String, policy: TurnPolicy, finalize: Bool, repair: Bool,
        send: (ConversationAction) -> Void
    ) async -> TurnEnd {
        guard let session = prepare() else {
            return TurnEnd(kind: .failed("No model is loaded."), outcome: "failed", answer: "")
        }
        begin(run: run, policy: policy, repair: repair, session: session)
        let text = finalize ? ConversationFeature.finalized(prompt) : prompt
        let options = GenerationOptions(maximumResponseTokens: policy.maxTokens)
        let context = ContextOptions(
            reasoningLevel: (AgentReasoningEffort(rawValue: policy.settings.effort) ?? .low).level)
        let (events, sink) = AsyncStream.makeStream(of: Event.self)
        self.sink = sink
        var failure: (any Error)?
        var stopped = false
        await withTaskGroup(of: Void.self) { group in
            group.addTask { [weak self] in await self?.observe(session) }
            group.addTask {
                do {
                    for try await _ in session.streamResponse(to: text, options: options, contextOptions: context) {}
                    sink.yield(.ended(nil))
                } catch {
                    sink.yield(.ended(error))
                }
            }
            loop: for await event in events {
                switch event {
                case .actions(let actions):
                    actions.forEach(send)
                case .stop:
                    stopped = true
                    group.cancelAll()
                case .ended(let error):
                    failure = error
                    break loop
                }
            }
            group.cancelAll()
        }
        self.sink = nil
        sink.finish()
        while session.isResponding, !Task.isCancelled { try? await Task.sleep(for: .milliseconds(10)) }
        tools.drain()
        let transcript = Array(session.transcript)
        progress(transcript).forEach(send)
        let end = await finish(transcript, failure: failure, stopped: stopped, send: send)
        if mode == .chat {
            let stats = generationStats(transcript, failure: failure, stopped: stopped)
            lastGeneration = stats
            MetricsLog.generation(model: modelName, stats: stats)
        }
        return end
    }

    func stop(run: Int) {
        guard run == self.run else { return }
        stopRequested = true
        sink?.yield(.stop)
    }

    func request(_ call: AgentToolCall) -> ToolCallRecord? {
        guard let session, let sink else { return nil }
        let transcript = Array(session.transcript)
        var found: ToolCallRecord?
        search: for entry in transcript.dropFirst(base).reversed() {
            guard case .toolCalls(let calls) = entry else { continue }
            for item in calls
            where !claimed.contains(item.id) && item.toolName == call.name
                && AgentPromptBuilder.arguments(item.toolName, item.arguments) == call.arguments
            {
                found = ToolCallRecord(id: item.id, name: item.toolName, arguments: item.arguments.jsonString)
                break search
            }
        }
        let record = found ?? ToolCallRecord(id: UUID().uuidString, name: call.name, arguments: Self.json(call))
        claimed.insert(record.id)
        let converted = Self.record(transcript)
        let done =
            prompt(in: transcript).map {
                converted.steps(from: $0 + 1).prefix { $0.outputs != nil }.count
            } ?? 0
        if !stopRequested, offset + done + 1 >= maxSteps { limitReached = true }
        sink.yield(.actions([.transcriptChanged(run: run, converted), .toolRequested(run: run, record)]))
        return record
    }

    func snapshotTokens() async {
        guard currentTokens == nil, let model else { return }
        currentTokens = []
        currentTokens = await model.processedTokens()
    }

    private func prepare() -> LanguageModelSession? {
        if let session { return session }
        guard let model = chat.model else { return nil }
        let date = preferences.fixedDate ?? AgentPromptBuilder.today()
        let session = LanguageModelSession(
            model: model, tools: tools.tools(fileOperations: preferences.fileOperations),
            instructions: AgentPromptBuilder.instructions(date: date, citeFetched: preferences.citeFetched))
        session.transcriptErrorHandlingPolicy = .preserveTranscript
        self.session = session
        self.model = model
        currentDate = date
        modelName = chat.modelName
        return session
    }

    private func begin(run: Int, policy: TurnPolicy, repair: Bool, session: LanguageModelSession) {
        self.run = run
        mode = policy.mode
        effort = policy.settings.effort
        maxSteps = policy.maxSteps
        offset = repair ? step : 0
        if !repair { turnIndex += 1 }
        base = session.transcript.count
        completed = 0
        limitReached = false
        stopRequested = false
        claimed = []
        turnMark = session.usage
        stepMark = session.usage
        started = .now
        stepStart = .now
        ttft = nil
        firstEntryAt = nil
        currentTokens = nil
    }

    private func observe(_ session: LanguageModelSession) async {
        for await transcript in Observations({ session.transcript }) {
            sink?.yield(.actions(progress(Array(transcript))))
        }
    }

    private func prompt(in transcript: [Transcript.Entry]) -> Int? {
        transcript.indices.dropFirst(base).first { if case .prompt = transcript[$0] { true } else { false } }
    }

    private func progress(_ transcript: [Transcript.Entry]) -> [ConversationAction] {
        let record = Self.record(transcript)
        var actions: [ConversationAction] = [.transcriptChanged(run: run, record)]
        guard let prompt = prompt(in: transcript) else { return actions }
        if firstEntryAt == nil, transcript.count > prompt + 1 { firstEntryAt = .now }
        let steps = record.steps(from: prompt + 1)
        while !limitReached, completed < steps.count, steps[completed].outputs != nil {
            actions.append(.stepFinished(run: run, finishStep(transcript, steps[completed].generated)))
            completed += 1
            stepStart = .now
            ttft = nil
            if step > maxSteps { limitReached = true }
        }
        if !limitReached, completed < steps.count, ttft == nil, !steps[completed].generated.isEmpty {
            ttft = (ContinuousClock.now - stepStart) / .seconds(1)
        }
        return actions
    }

    private func finish(
        _ transcript: [Transcript.Entry], failure: (any Error)?, stopped: Bool, send: (ConversationAction) -> Void
    ) async -> TurnEnd {
        let steps = prompt(in: transcript).map { Self.record(transcript).steps(from: $0 + 1) } ?? []
        let current = completed < steps.count ? steps[completed].generated : transcript.count..<transcript.count
        let partial = Self.responseText(transcript[min(base, transcript.count)...])
        if stopped {
            return TurnEnd(kind: .stopped, outcome: limitReached ? "step-limit" : "stopped", answer: partial)
        }
        if let failure {
            guard case LanguageModelError.contextSizeExceeded(let exceeded) = failure else {
                return TurnEnd(kind: .failed(String(describing: failure)), outcome: "failed", answer: partial)
            }
            let message =
                "\(ConversationFeature.contextFullPrefix): the conversation needs \(exceeded.tokenCount) tokens and the limit is "
                + "\(exceeded.contextSize). Start a new conversation, lower the page budget or ask for something narrower."
            return TurnEnd(kind: .failed(message), outcome: "overflow", answer: partial)
        }
        await snapshotTokens()
        let answer = Self.responseText(transcript[current])
        let citations =
            mode == .work && !answer.isEmpty
            ? await CitationGuard.checkCitations(answer, pages: tools.seenPages) : nil
        let stats = finishStep(transcript, current, citations: citations)
        send(.stepFinished(run: run, stats))
        if let citations {
            send(.guardNote(run: run, title: CitationGuard.citationsTitle, text: citations.note, warning: false))
        }
        let stopReason = try? Self.metadata(transcript[current])["stopReason"]?.value(String.self)
        let outcome =
            answer.isEmpty
            ? "failed" : stats.finish == "eos" ? TurnEnd.answered : stats.finish == "cap" ? "cap" : "failed"
        return TurnEnd(kind: .ended(stopReason: stopReason), outcome: outcome, answer: answer)
    }

    private func finishStep(
        _ transcript: [Transcript.Entry], _ range: Range<Int>, citations: CitationCheck? = nil
    ) -> StepStats {
        let usage = session?.usage ?? stepMark
        let input = (usage?.input.totalTokenCount ?? 0) - (stepMark?.input.totalTokenCount ?? 0)
        let reused = (usage?.input.cachedTokenCount ?? 0) - (stepMark?.input.cachedTokenCount ?? 0)
        let output = (usage?.output.totalTokenCount ?? 0) - (stepMark?.output.totalTokenCount ?? 0)
        stepMark = usage
        let entries = transcript[range]
        let metadata = Self.metadata(entries)
        let stopReason = (try? metadata["stopReason"]?.value(String.self)) ?? "unknown"
        let finish = stopReason == "stopToken" ? "eos" : stopReason == "maxTokens" ? "cap" : stopReason
        let shared = try? metadata["sharedTokens"]?.value(Int.self)
        let truncated = (try? metadata["truncatedToolOutputs"]?.value(Int.self)) ?? 0
        let prefill = try? metadata["prefillSeconds"]?.value(Double.self)
        let decode = try? metadata["decodeSeconds"]?.value(Double.self)
        let speed = decode.flatMap { $0 > 0 ? Double(output) / $0 : nil }
        let calls = entries.flatMap { entry -> [String] in
            if case .toolCalls(let calls) = entry { calls.map(\.toolName) } else { [] }
        }
        let thinking = entries.map {
            if case .reasoning(let reasoning) = $0 { Self.trimmed(reasoning.segments).count } else { 0 }
        }.reduce(0, +)
        let answer = Self.responseText(entries)

        var diverged = false
        if mode == .work, let previous, let shared, shared < previous.input + previous.output - 1 {
            diverged = true
            divergeCount += 1
            var tokens = ""
            if let before = previousTokens, let after = currentTokens, shared < min(before.count, after.count) {
                tokens =
                    " generated \(Array(before[max(0, shared - 2)..<min(before.count, shared + 4)]))"
                    + " re-rendered \(Array(after[max(0, shared - 2)..<min(after.count, shared + 4)]))"
            }
            AgentTools.log(
                "[agent] re-render diverges at step \(step): shared \(shared) < previous prompt \(previous.input) + output \(previous.output) - 1"
                    + tokens)
        }
        previous = (input, output)
        previousTokens = currentTokens
        currentTokens = nil
        if finish == "cap" { capSteps += 1 }
        lastFinish = finish

        if mode == .work {
            var record = MetricsLog.agent("message", model: modelName)
            record.agentSession = id
            record.agentTurn = turnIndex
            record.agentStep = step
            record.agentEffort = effort
            record.promptTokens = input
            record.reusedTokens = reused
            record.generatedTokens = output
            record.ttftSeconds = ttft
            record.prefillSeconds = prefill
            record.decodeTokPerSec = speed
            record.finishReason = finish
            record.rssMB = MetricsLog.residentMB()
            record.agentToolCalls = calls
            record.agentRerenderDiverged = diverged
            record.agentThinkingChars = thinking
            record.agentAnswerChars = answer.count
            record.truncatedToolOutputs = truncated
            record.splicedGenerations = try? metadata["splicedGenerations"]?.value(Int.self)
            record.citationsTotal = citations?.total
            record.citationsSeen = citations?.seenCount
            record.citationMeanOverlap = citations?.meanOverlap
            MetricsLog.save(record)
        }
        return StepStats(
            step: step, ttftSeconds: ttft, tokensPerSecond: speed, promptTokens: input, reusedTokens: reused,
            generatedTokens: output, toolCalls: calls.count, finish: finish)
    }

    private func generationStats(_ transcript: [Transcript.Entry], failure: (any Error)?, stopped: Bool)
        -> GenerationStats
    {
        let added = transcript.dropFirst(base)
        let after = session?.usage
        var stats = GenerationStats()
        stats.turn = transcript.count { if case .prompt = $0 { true } else { false } }
        stats.answer = added.map { if case .response(let response) = $0 { Self.text(response.segments) } else { "" } }
            .joined()
        stats.outputTokens = (after?.output.totalTokenCount ?? 0) - (turnMark?.output.totalTokenCount ?? 0)
        stats.ttftSeconds = firstEntryAt.map { ($0 - started) / .seconds(1) }
        if stopped {
            stats.stopReason = "cancelled"
        } else if let failure {
            if case LanguageModelError.contextSizeExceeded = failure {
                stats.stopReason = "contextSizeExceeded"
            } else {
                stats.stopReason = "error"
                stats.error = failure.localizedDescription
            }
        } else {
            var metadata: [String: GeneratedContent] = [:]
            if case .response(let response)? = added.last(where: { if case .response = $0 { true } else { false } }) {
                metadata = response.metadata
            }
            stats.inputTokens = (after?.input.totalTokenCount ?? 0) - (turnMark?.input.totalTokenCount ?? 0)
            stats.cachedTokens = (after?.input.cachedTokenCount ?? 0) - (turnMark?.input.cachedTokenCount ?? 0)
            stats.stopReason = (try? metadata["stopReason"]?.value(String.self)) ?? "unknown"
            if let decode = try? metadata["decodeSeconds"]?.value(Double.self), decode > 0 {
                stats.tokensPerSecond = Double(stats.outputTokens) / decode
            }
        }
        return stats
    }

    static func record(_ transcript: [Transcript.Entry]) -> TranscriptRecord {
        TranscriptRecord(
            entries: transcript.map { entry in
                switch entry {
                case .instructions(let value):
                    .instructions(id: value.id, text: text(value.segments))
                case .prompt(let value):
                    .prompt(id: value.id, text: text(value.segments))
                case .reasoning(let value):
                    .reasoning(id: value.id, text: text(value.segments))
                case .response(let value):
                    .response(id: value.id, text: text(value.segments), metadata: stepMetadata(value.metadata))
                case .toolCalls(let calls):
                    .toolCalls(
                        id: calls.id,
                        calls: calls.map {
                            ToolCallRecord(id: $0.id, name: $0.toolName, arguments: $0.arguments.jsonString)
                        },
                        metadata: stepMetadata(calls.last?.metadata ?? [:]))
                case .toolOutput(let output):
                    .toolOutput(id: output.id, name: output.toolName, text: text(output.segments))
                @unknown default:
                    .response(id: entry.id, text: "", metadata: nil)
                }
            })
    }

    private static func stepMetadata(_ metadata: [String: GeneratedContent]) -> StepMetadata? {
        guard !metadata.isEmpty else { return nil }
        return StepMetadata(
            stopReason: try? metadata["stopReason"]?.value(String.self),
            sharedTokens: try? metadata["sharedTokens"]?.value(Int.self),
            truncatedToolOutputs: try? metadata["truncatedToolOutputs"]?.value(Int.self),
            prefillSeconds: try? metadata["prefillSeconds"]?.value(Double.self),
            decodeSeconds: try? metadata["decodeSeconds"]?.value(Double.self),
            reservedTokens: try? metadata["reservedTokens"]?.value(Int.self))
    }

    private static func metadata(_ entries: ArraySlice<Transcript.Entry>) -> [String: GeneratedContent] {
        for entry in entries.reversed() {
            switch entry {
            case .response(let response) where !response.metadata.isEmpty: return response.metadata
            case .toolCalls(let calls) where !(calls.last?.metadata.isEmpty ?? true): return calls.last?.metadata ?? [:]
            default: continue
            }
        }
        return [:]
    }

    private static func json(_ call: AgentToolCall) -> String {
        let object = Dictionary(call.arguments.map { ($0.name, $0.value) }, uniquingKeysWith: { first, _ in first })
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) else {
            return "{}"
        }
        return String(decoding: data, as: UTF8.self)
    }

    static func responseText(_ entries: ArraySlice<Transcript.Entry>) -> String {
        entries.map { if case .response(let response) = $0 { text(response.segments) } else { "" } }
            .joined().trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func text(_ segments: [Transcript.Segment]) -> String {
        segments.map { if case .text(let text) = $0 { text.content } else { "" } }.joined()
    }

    private static func trimmed(_ segments: [Transcript.Segment]) -> String {
        text(segments).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
