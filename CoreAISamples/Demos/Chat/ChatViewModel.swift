import CoreAIBackend
import Foundation
import FoundationModels
import Observation
import RAGContext

struct GenerationStats: Encodable, Sendable {
    var turn = 0
    var answer = ""
    var inputTokens: Int?
    var cachedTokens: Int?
    var outputTokens = 0
    var ttftSeconds: Double?
    var tokensPerSecond: Double?
    var stopReason = ""
    var error: String?
    var sources: Int?
    var sourceTokens: Int?
    var ragPromptTokens: Int?
    var retrievalSeconds: Double?
}

@MainActor
@Observable
final class ChatViewModel {
    enum Phase: Equatable {
        case empty
        case loading(preparing: Bool)
        case ready
        case failed(String)
    }

    struct Turn: Identifiable {
        let id: String
        let prompt: String
        var reasoning = ""
        var response = ""
        var notes: [String] = []
        var sources: [String] = []
    }

    static let maxTokenChoices = [256, 512, 1024, 2048, 4096, 8192]
    static let defaultContextLimit = 16_384

    private(set) var phase = Phase.empty
    private(set) var bundleURL: URL?
    private(set) var modelName = ""
    private(set) var loadNote = ""
    private(set) var loadStarted = Date.now
    private(set) var loadSeconds = 0.0
    private(set) var entries: [Transcript.Entry] = []
    private(set) var isResponding = false
    private(set) var last: GenerationStats?
    private(set) var contextNotice: String?
    private(set) var contextFull = false
    private(set) var retrieving = false
    private(set) var retrieval: RAGRetrieval?
    private(set) var ragTurns: [(after: Int, turn: Turn)] = []
    var input = ""
    var externallyBusy = false
    var reasoning = AgentReasoningEffort.medium
    var maxTokens = 2048
    var ragEnabled = false
    var ragWeb = true
    var ragLocal = true
    var ragBudget = ChatViewModel.launchRAGBudget ?? RAGContext.sourceTokenBudget
    let contextLimit = LaunchArguments.value("--context-limit").flatMap(Int.init) ?? ChatViewModel.defaultContextLimit

    @ObservationIgnored private(set) var model: LocalLanguageModel?
    @ObservationIgnored private(set) var generation: Task<Void, Never>?
    @ObservationIgnored private var session: LanguageModelSession?
    @ObservationIgnored private var consumer: Task<(any Error)?, Never>?
    @ObservationIgnored private var firstEntryAt: ContinuousClock.Instant?
    @ObservationIgnored private var run = 0

    nonisolated static var launchRAGBudget: Int? {
        LaunchArguments.value("--rag-budget").flatMap(Int.init).flatMap {
            RAGSettings.budgetChoices.contains($0) ? $0 : nil
        }
    }

    var isLoading: Bool { if case .loading = phase { true } else { false } }
    var trimmedInput: String { input.trimmingCharacters(in: .whitespacesAndNewlines) }
    var canSend: Bool { phase == .ready && !isResponding && !externallyBusy && !contextFull && !trimmedInput.isEmpty }
    var loadStats: Engine.LoadStats? { model?.loadStats }
    var ragIsOn: Bool { ragEnabled && (ragWeb || ragLocal) }

    var turns: [Turn] {
        var turns = Self.turns(entries)
        for rag in ragTurns.reversed() { turns.insert(rag.turn, at: min(rag.after, turns.count)) }
        return turns
    }

    static func turns(_ entries: some Sequence<Transcript.Entry>) -> [Turn] {
        var turns: [Turn] = []
        for entry in entries {
            switch entry {
            case .prompt(let prompt):
                turns.append(Turn(id: entry.id, prompt: Self.text(prompt.segments)))
            case .reasoning(let reasoning) where !turns.isEmpty:
                turns[turns.count - 1].reasoning += Self.text(reasoning.segments)
            case .response(let response) where !turns.isEmpty:
                turns[turns.count - 1].response += Self.text(response.segments)
            default:
                break
            }
        }
        return turns
    }

    var statusLine: String {
        guard let last else { return loadNote }
        if let error = last.error { return "Error: \(error)" }
        var parts: [String] = []
        if let ttft = last.ttftSeconds { parts.append(String(format: "TTFT %.2f s", ttft)) }
        if let speed = last.tokensPerSecond { parts.append(String(format: "%.1f tok/s", speed)) }
        if let input = last.inputTokens { parts.append("input \(input)") }
        if let cached = last.cachedTokens { parts.append("reused \(cached)") }
        parts.append("output \(last.outputTokens)")
        parts.append(last.stopReason == "cancelled" ? "stopped" : last.stopReason)
        if let sources = last.sources { parts.append("sources \(sources)") }
        if let tokens = last.sourceTokens { parts.append("source tok \(tokens)") }
        if let tokens = last.ragPromptTokens { parts.append("prompt(rag) \(tokens) tok") }
        if let seconds = last.retrievalSeconds { parts.append(String(format: "retrieval %.1fs", seconds)) }
        return parts.joined(separator: "  |  ")
    }

    func isUsing(_ url: URL) -> Bool {
        phase != .empty && bundleURL.map(ModelStorage.key) == ModelStorage.key(url)
    }

    func state(of url: URL) -> String? {
        guard isUsing(url) else { return nil }
        switch phase {
        case .ready: return "Loaded"
        case .loading(let preparing): return preparing ? "Preparing…" : "Loading…"
        default: return nil
        }
    }

    func load(_ url: URL) async {
        guard !isLoading, !isResponding, !externallyBusy else { return }
        let key = ModelStorage.key(url)
        model = nil
        session = nil
        entries = []
        ragTurns = []
        last = nil
        contextNotice = nil
        contextFull = false
        bundleURL = url
        modelName = (try? ModelBundle(url: url).name) ?? url.lastPathComponent
        loadStarted = .now
        phase = .loading(preparing: !ModelStorage.prepared.contains(key))
        let clock = ContinuousClock()
        let start = clock.now
        do {
            _ = try BundleSupport.supportedBundle(at: url)
            let model = try await LocalLanguageModel(
                bundle: url, contextLimit: contextLimit, overflow: .dropOldToolOutputs)
            loadSeconds = (clock.now - start) / .seconds(1)
            self.model = model
            ModelStorage.prepared.insert(key)
            loadNote =
                model.loadStats.cached
                ? String(format: "Loaded in %.1f s", loadSeconds)
                : String(
                    format: "Prepared for this Mac in %.1f s, loaded in %.1f s", model.loadStats.specializeSeconds,
                    loadSeconds)
            phase = .ready
            newChat()
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func resetRAG() {
        ragTurns = []
        retrieval = nil
        last = nil
        contextNotice = nil
        contextFull = false
    }

    func newChat() {
        guard let model, !isResponding else { return }
        let session = LanguageModelSession(model: model)
        session.transcriptErrorHandlingPolicy = .preserveTranscript
        self.session = session
        entries = []
        ragTurns = []
        retrieval = nil
        last = nil
        contextNotice = nil
        contextFull = false
    }

    @discardableResult
    func send() -> Bool {
        guard canSend, let session, let model else { return false }
        let prompt = trimmedInput
        input = ""
        last = nil
        contextNotice = nil
        isResponding = true
        let options = GenerationOptions(maximumResponseTokens: maxTokens)
        let context = ContextOptions(reasoningLevel: reasoning.level)
        if ragIsOn {
            var settings = RAGSettings.fromDefaults()
            settings.web = ragWeb
            settings.local = ragLocal
            settings.sourceTokenBudget = ragBudget
            generation = Task { await respondFromSources(model, prompt, settings, options, context) }
        } else {
            generation = Task { await respond(session, prompt, options, context) }
        }
        return true
    }

    func stop() {
        if retrieving { generation?.cancel() } else { consumer?.cancel() }
    }

    private func respondFromSources(
        _ model: LocalLanguageModel, _ question: String, _ settings: RAGSettings, _ options: GenerationOptions,
        _ context: ContextOptions
    ) async {
        ragTurns.append((Self.turns(entries).count, Turn(id: UUID().uuidString, prompt: question)))
        retrieving = true
        let today = RAGContext.today()
        let retrieval = await RAGRetriever.gather(question: question, settings: settings, count: model.countTokens)
        let promptTokens = model.countTokens(
            RAGContext.prompt(question: question, sources: retrieval.sources, today: today))
        retrieving = false
        self.retrieval = retrieval
        ragTurns[ragTurns.count - 1].turn.notes = retrieval.notes
        ragTurns[ragTurns.count - 1].turn.sources = retrieval.citations
        for note in retrieval.notes { AgentTools.log("[rag] " + note) }
        var stats = GenerationStats(
            turn: ragTurns.count, sources: retrieval.sources.count, sourceTokens: retrieval.sourceTokens,
            retrievalSeconds: retrieval.seconds)
        guard !Task.isCancelled, !retrieval.sources.isEmpty else {
            if Task.isCancelled {
                stats.stopReason = "cancelled"
            } else {
                stats.stopReason = "noSources"
                stats.answer = RAGContext.noSourcesReply
                ragTurns[ragTurns.count - 1].turn.response = RAGContext.noSourcesReply
            }
            last = stats
            generation = nil
            isResponding = false
            return
        }
        stats.ragPromptTokens = promptTokens
        let session = LanguageModelSession(model: model, instructions: RAGContext.instruction(today: today))
        session.transcriptErrorHandlingPolicy = .preserveTranscript
        await respond(
            session, RAGContext.message(question: question, sources: retrieval.sources), options, context, rag: stats)
    }

    private func show(_ transcript: Transcript, rag: Bool) {
        guard rag else {
            entries = Array(transcript)
            return
        }
        guard let live = Self.turns(transcript).last, !ragTurns.isEmpty else { return }
        ragTurns[ragTurns.count - 1].turn.reasoning = live.reasoning
        ragTurns[ragTurns.count - 1].turn.response = live.response
    }

    private func respond(
        _ session: LanguageModelSession, _ prompt: String, _ options: GenerationOptions, _ context: ContextOptions,
        rag: GenerationStats? = nil
    ) async {
        run += 1
        let id = run
        let base = session.transcript.count
        let before = session.usage
        let start = ContinuousClock.now
        firstEntryAt = nil
        let observer = Task { [weak self] in
            for await transcript in Observations({ session.transcript }) {
                guard let self, self.run == id, self.isResponding else { return }
                if self.firstEntryAt == nil, transcript.count > base + 1 { self.firstEntryAt = .now }
                self.show(transcript, rag: rag != nil)
            }
        }
        let consumer = Task { () -> (any Error)? in
            do {
                for try await _ in session.streamResponse(to: prompt, options: options, contextOptions: context) {}
                return nil
            } catch {
                return error
            }
        }
        self.consumer = consumer
        let failure = await consumer.value
        let stopped = consumer.isCancelled
        while session.isResponding { try? await Task.sleep(for: .milliseconds(10)) }
        observer.cancel()
        show(session.transcript, rag: rag != nil)

        let added = session.transcript.dropFirst(base)
        let after = session.usage
        var stats = rag ?? GenerationStats()
        if rag == nil { stats.turn = session.transcript.count { if case .prompt = $0 { true } else { false } } }
        stats.answer = added.map { if case .response(let response) = $0 { Self.text(response.segments) } else { "" } }
            .joined()
        stats.outputTokens = after.output.totalTokenCount - before.output.totalTokenCount
        stats.ttftSeconds = firstEntryAt.map { ($0 - start) / .seconds(1) }
        if stopped {
            stats.stopReason = "cancelled"
        } else if let failure {
            if case LanguageModelError.contextSizeExceeded(let exceeded) = failure {
                stats.stopReason = "contextSizeExceeded"
                contextFull = true
                contextNotice =
                    "The context window is full: this chat needs \(exceeded.tokenCount) tokens and the limit is \(exceeded.contextSize). Start a New Chat to continue."
            } else {
                stats.stopReason = "error"
                stats.error = failure.localizedDescription
            }
        } else {
            var metadata: [String: GeneratedContent] = [:]
            if case .response(let response)? = added.last(where: { if case .response = $0 { true } else { false } }) {
                metadata = response.metadata
            }
            stats.inputTokens = after.input.totalTokenCount - before.input.totalTokenCount
            stats.cachedTokens = after.input.cachedTokenCount - before.input.cachedTokenCount
            stats.stopReason = (try? metadata["stopReason"]?.value(String.self)) ?? "unknown"
            if let decode = try? metadata["decodeSeconds"]?.value(Double.self), decode > 0 {
                stats.tokensPerSecond = Double(stats.outputTokens) / decode
            }
            if stats.stopReason == "maxTokens", let reserved = try? metadata["reservedTokens"]?.value(Int.self),
                reserved < options.maximumResponseTokens ?? .max
            {
                contextNotice =
                    "The reply stopped at the end of the context window (\(reserved) tokens were left for it). Start a New Chat if the next message does not fit."
            }
        }
        last = stats
        MetricsLog.generation(model: modelName, stats: stats)
        self.consumer = nil
        generation = nil
        isResponding = false
    }

    static func text(_ segments: [Transcript.Segment]) -> String {
        segments.map { if case .text(let text) = $0 { text.content } else { "" } }.joined()
    }
}
