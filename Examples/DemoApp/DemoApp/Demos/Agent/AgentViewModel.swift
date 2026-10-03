import AppKit
import CoreAIBackend
import Foundation
import FoundationModels
import Observation

@MainActor
@Observable
final class AgentViewModel {
    struct Entry: Identifiable {
        enum Kind { case user, thinking, answer, toolCall, toolResult, failure, note }
        let id: String
        var kind: Kind
        var title: String
        var text: String
        var stats: String?
        var live = false

        init(id: String = UUID().uuidString, kind: Kind, title: String, text: String, stats: String? = nil, live: Bool = false) {
            self.id = id
            self.kind = kind
            self.title = title
            self.text = text
            self.stats = stats
            self.live = live
        }
    }

    private struct Run {
        let base: Int
        var mark: LanguageModelSession.Usage
        var stepStart = ContinuousClock.now
        var ttft: Double?
        var groups = 0
        var limitReached = false
    }

    static let effortDefaultsKey = "agent.reasoningEffort"
    static let maxStepsDefaultsKey = "agent.maxSteps"
    static let maxTokensDefaultsKey = "agent.maxTokens"
    static let pageBudgetDefaultsKey = "agent.pageBudget"
    static let braveKeyDefaultsKey = "agent.braveAPIKey"
    static let fileOpsDefaultsKey = "agent.allowFileOps"
    static let notesFolderDefaultsKey = "agent.notesFolder"
    static let maxTokenChoices = [512, 1024, 1536, 2048, 4096, 8192]
    nonisolated static let noAnswerText = "(no answer text)"
    static let contextFullNote = "Context full — dropped the oldest tool output"

    var persistsSettings = true
    var effort: AgentReasoningEffort { didSet { store(effort.rawValue, Self.effortDefaultsKey) } }
    var maxIterations: Int { didSet { store(maxIterations, Self.maxStepsDefaultsKey) } }
    var maxNewTokens: Int { didSet { store(maxNewTokens, Self.maxTokensDefaultsKey) } }
    var pageCharacterBudget: Int { didSet { store(pageCharacterBudget, Self.pageBudgetDefaultsKey) } }
    var braveAPIKey: String { didSet { store(braveAPIKey, Self.braveKeyDefaultsKey) } }
    var allowFileOperations: Bool { didSet { store(allowFileOperations, Self.fileOpsDefaultsKey) } }
    var notesFolderPath: String { didSet { store(notesFolderPath, Self.notesFolderDefaultsKey) } }
    var input = ""
    var entries: [Entry] = []

    private(set) var statusLine = ""
    private(set) var running = false
    private(set) var divergeCount = 0
    private(set) var capSteps = 0
    private(set) var lastFinishReason: String?
    private(set) var sessionID: String?
    private(set) var currentDate: String?
    private(set) var modelName = ""

    @ObservationIgnored private var session: LanguageModelSession?
    @ObservationIgnored private var model: LocalLanguageModel?
    @ObservationIgnored private var sessionLoad: Date?
    @ObservationIgnored private var previousTokens: [Int32]?
    @ObservationIgnored private var currentTokens: [Int32]?
    @ObservationIgnored private var consumer: Task<(any Error)?, Never>?
    @ObservationIgnored private var run: Run?
    @ObservationIgnored private var extras: [(after: Int, entry: Entry)] = []
    @ObservationIgnored private var hidden: Set<String> = []
    @ObservationIgnored private var stats: [String: String] = [:]
    @ObservationIgnored private var seenPages: [String: String] = [:]
    @ObservationIgnored private var turnIndex = 0
    @ObservationIgnored private var step = 0
    @ObservationIgnored private var previous: (input: Int, output: Int)?

    init() {
        let defaults = UserDefaults.standard
        func positive(_ key: String) -> Int? { (defaults.object(forKey: key) as? Int).flatMap { $0 > 0 ? $0 : nil } }
        effort = defaults.string(forKey: Self.effortDefaultsKey).flatMap(AgentReasoningEffort.init(rawValue:)) ?? .low
        maxIterations = positive(Self.maxStepsDefaultsKey) ?? 8
        maxNewTokens = positive(Self.maxTokensDefaultsKey) ?? 1_536
        pageCharacterBudget = positive(Self.pageBudgetDefaultsKey) ?? 10_000
        braveAPIKey = defaults.string(forKey: Self.braveKeyDefaultsKey) ?? ""
        allowFileOperations = defaults.bool(forKey: Self.fileOpsDefaultsKey)
        notesFolderPath = defaults.string(forKey: Self.notesFolderDefaultsKey) ?? ""
    }

    private func store(_ value: Any, _ key: String) {
        guard persistsSettings else { return }
        UserDefaults.standard.set(value, forKey: key)
    }

    var trimmedInput: String { input.trimmingCharacters(in: .whitespacesAndNewlines) }
    var canReset: Bool { !running && !entries.isEmpty }
    var searchProvider: String { braveAPIKey.isEmpty ? "DuckDuckGo" : "Brave Search API" }

    var notesDirectory: URL {
        notesFolderPath.isEmpty
            ? AgentTools.defaultNotesDirectory()
            : URL(fileURLWithPath: (notesFolderPath as NSString).expandingTildeInPath, isDirectory: true)
    }

    var toolSettings: AgentToolSettings {
        AgentToolSettings(
            pageCharacterBudget: pageCharacterBudget, braveAPIKey: braveAPIKey, notesDirectory: notesDirectory,
            allowFileOperations: allowFileOperations)
    }

    func canSend(chat: ChatViewModel) -> Bool {
        !running && !trimmedInput.isEmpty && chat.phase == .ready && !chat.isResponding
    }

    @discardableResult
    func send(chat: ChatViewModel) -> Bool {
        guard canSend(chat: chat), let model = chat.model else { return false }
        if sessionLoad != chat.loadStarted { reset() }
        let text = trimmedInput
        input = ""
        if session == nil {
            let date = AgentPromptBuilder.today()
            let session = LanguageModelSession(
                model: model, tools: AgentPromptBuilder.tools(fileOperations: allowFileOperations, run: toolRunner()),
                instructions: AgentPromptBuilder.instructions(date: date))
            session.transcriptErrorHandlingPolicy = .preserveTranscript
            self.session = session
            self.model = model
            sessionLoad = chat.loadStarted
            sessionID = Self.newSessionID()
            currentDate = date
        }
        guard let session else { return false }
        modelName = chat.modelName
        turnIndex += 1
        running = true
        lastFinishReason = nil
        chat.externallyBusy = true
        statusLine = "Step 1 of \(maxIterations): generating…"
        let options = GenerationOptions(maximumResponseTokens: maxNewTokens)
        let context = ContextOptions(reasoningLevel: effort.level)
        Task {
            await respond(session, text, options, context)
            chat.externallyBusy = false
            saveTranscript()
        }
        return true
    }

    func stop() { consumer?.cancel() }

    func reset() {
        guard !running else { return }
        session = nil
        model = nil
        sessionLoad = nil
        previousTokens = nil
        currentTokens = nil
        sessionID = nil
        currentDate = nil
        entries = []
        extras = []
        hidden = []
        stats = [:]
        seenPages = [:]
        divergeCount = 0
        capSteps = 0
        turnIndex = 0
        previous = nil
        lastFinishReason = nil
        statusLine = ""
    }

    private func toolRunner() -> @Sendable (AgentToolCall) async -> String {
        { [weak self] call in
            guard let settings = await self?.toolSettings else { return "error: the agent is no longer running" }
            await self?.snapshotTokens()
            let start = Date()
            let output = await AgentTools.run(call, settings: settings)
            await self?.recordTool(call, output, start: start, end: Date())
            return output.text
        }
    }

    private func snapshotTokens() async {
        guard currentTokens == nil, let model else { return }
        currentTokens = []
        currentTokens = await model.processedTokens()
    }

    private func recordTool(_ call: AgentToolCall, _ output: AgentToolOutput, start: Date, end: Date) {
        seenPages.merge(output.seen) { _, new in new }
        let style = Date.ISO8601FormatStyle(includingFractionalSeconds: true)
        var record = MetricsLog.agent("agent-tool", model: modelName)
        record.agentSession = sessionID
        record.agentTurn = turnIndex
        record.agentStep = step
        record.agentEffort = effort.rawValue
        record.toolName = call.name
        record.toolArg = (call.value("url") ?? call.value("query") ?? call.value("title")).map { String($0.prefix(200)) }
        record.toolStart = start.ISO8601Format(style)
        record.toolEnd = end.ISO8601Format(style)
        record.toolSeconds = end.timeIntervalSince(start)
        record.toolResultChars = output.text.count
        record.toolError = output.text.hasPrefix("error:")
        MetricsLog.save(record)
    }

    private func respond(_ session: LanguageModelSession, _ prompt: String, _ options: GenerationOptions, _ context: ContextOptions) async {
        run = Run(base: session.transcript.count, mark: session.usage)
        step = 1
        currentTokens = nil
        let observer = Task { [weak self] in
            for await transcript in Observations({ session.transcript }) {
                guard let self, self.run != nil else { return }
                self.observe(Array(transcript), session: session, live: true)
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
        let transcript = Array(session.transcript)
        observe(transcript, session: session, live: false)
        await finish(transcript, session: session, failure: failure, stopped: stopped)
        run = nil
        self.consumer = nil
        running = false
    }

    private func observe(_ transcript: [Transcript.Entry], session: LanguageModelSession, live: Bool) {
        guard var run else { return }
        let steps = Self.steps(transcript, from: run.base + 1)
        while !run.limitReached, run.groups < steps.count, steps[run.groups].outputs != nil {
            finishStep(transcript, steps[run.groups].generated, session: session, ttft: run.ttft, mark: &run.mark)
            run.groups += 1
            run.stepStart = .now
            run.ttft = nil
            step = run.groups + 1
            if step > maxIterations {
                run.limitReached = true
                consumer?.cancel()
            } else {
                statusLine = "Step \(step) of \(maxIterations): generating…"
            }
        }
        if !run.limitReached, run.groups < steps.count {
            let current = transcript[steps[run.groups].generated]
            if run.ttft == nil, !current.isEmpty { run.ttft = (ContinuousClock.now - run.stepStart) / .seconds(1) }
            if live, let names = Self.pendingCalls(current) { statusLine = "Step \(step): running " + names.joined(separator: " + ") + "…" }
        }
        self.run = run
        rebuild(transcript, live: live)
    }

    private func finish(_ transcript: [Transcript.Entry], session: LanguageModelSession, failure: (any Error)?, stopped: Bool) async {
        guard var run else { return }
        let steps = Self.steps(transcript, from: run.base + 1)
        let current = run.groups < steps.count ? steps[run.groups].generated : transcript.count..<transcript.count
        let end = transcript.count
        if run.limitReached {
            hidden.formUnion(transcript[current].map(\.id))
            extras.append((end, Entry(
                kind: .failure, title: "Step limit reached",
                text: "The agent used all \(maxIterations) steps without finishing. Raise the limit in Settings or ask for something narrower.")))
            statusLine = "Stopped at the \(maxIterations)-step limit."
        } else if stopped {
            let partial = Self.responseText(transcript[current])
            hidden.formUnion(transcript[current].filter { if case .response = $0 { true } else { false } }.map(\.id))
            extras.append((end, Entry(kind: .failure, title: "Stopped", text: partial.isEmpty ? "(stopped before any answer)" : partial)))
            statusLine = "Stopped after \(step) step(s)."
        } else if let failure {
            if case LanguageModelError.contextSizeExceeded(let exceeded) = failure {
                extras.append((end, Entry(
                    kind: .failure, title: "Context window is full",
                    text: "The conversation needs \(exceeded.tokenCount) tokens but this bundle only has \(exceeded.contextSize). Lower the page budget or ask for something narrower.")))
                statusLine = "Context overflow at step \(step)."
            } else {
                extras.append((end, Entry(kind: .failure, title: "Generation failed", text: String(describing: failure))))
                statusLine = "Failed at step \(step)."
            }
        } else {
            rebuild(transcript, live: false)
            await snapshotTokens()
            let answer = Self.responseText(transcript[current])
            let citations = answer.isEmpty ? nil : await checkCitations(answer)
            finishStep(transcript, current, session: session, ttft: run.ttft, final: true, citations: citations, mark: &run.mark)
        }
        rebuild(transcript, live: false)
    }

    private func finishStep(
        _ transcript: [Transcript.Entry], _ range: Range<Int>, session: LanguageModelSession, ttft: Double?, final: Bool = false,
        citations: CitationCheck? = nil, mark: inout LanguageModelSession.Usage
    ) {
        let usage = session.usage
        let input = usage.input.totalTokenCount - mark.input.totalTokenCount
        let reused = usage.input.cachedTokenCount - mark.input.cachedTokenCount
        let output = usage.output.totalTokenCount - mark.output.totalTokenCount
        mark = usage
        let entries = transcript[range]
        let metadata = Self.metadata(entries)
        let stopReason = (try? metadata["stopReason"]?.value(String.self)) ?? "unknown"
        let finish = stopReason == "stopToken" ? "eos" : stopReason == "maxTokens" ? "cap" : stopReason
        let shared = try? metadata["sharedTokens"]?.value(Int.self)
        let truncated = (try? metadata["truncatedToolOutputs"]?.value(Int.self)) ?? 0
        let prefill = try? metadata["prefillSeconds"]?.value(Double.self)
        let decode = try? metadata["decodeSeconds"]?.value(Double.self)
        let speed = decode.flatMap { $0 > 0 ? Double(output) / $0 : nil }
        let calls = entries.flatMap { entry -> [String] in if case .toolCalls(let calls) = entry { calls.map(\.toolName) } else { [] } }
        let thinking = entries.map { if case .reasoning(let reasoning) = $0 { Self.trimmed(reasoning.segments).count } else { 0 } }.reduce(0, +)
        let answer = Self.responseText(entries)

        var diverged = false
        if let previous, let shared, shared < previous.input + previous.output - 1 {
            diverged = true
            divergeCount += 1
            var tokens = ""
            if let before = previousTokens, let after = currentTokens, shared < min(before.count, after.count) {
                tokens = " generated \(Array(before[max(0, shared - 2)..<min(before.count, shared + 4)]))"
                    + " re-rendered \(Array(after[max(0, shared - 2)..<min(after.count, shared + 4)]))"
            }
            AgentTools.log("[agent] re-render diverges at step \(step): shared \(shared) < previous prompt \(previous.input) + output \(previous.output) - 1" + tokens)
        }
        previous = (input, output)
        previousTokens = currentTokens
        currentTokens = nil
        if finish == "cap" { capSteps += 1 }
        lastFinishReason = finish

        let line = [
            "step \(step)",
            ttft.map { String(format: "TTFT %.2fs", $0) } ?? "TTFT n/a",
            String(format: "%.1f tok/s", speed ?? 0),
            "prompt \(input)",
            "reused \(reused)",
            "\(output) tok",
            calls.isEmpty ? "answered" : "\(calls.count) tool call(s)",
            finish,
        ].joined(separator: "  |  ")
        statusLine = line
        if truncated > 0 {
            extras.append((range.lowerBound, Entry(
                kind: .note, title: Self.contextFullNote, text: "\(truncated) tool output(s) shortened to fit the context window.")))
        }
        if final, answer.isEmpty {
            let placeholder = Entry(kind: .answer, title: "Assistant", text: Self.noAnswerText)
            extras.append((range.upperBound, placeholder))
            stats[placeholder.id] = line
        } else if let last = Self.lastDisplayID(entries) {
            stats[last] = line
        }
        if final, let citations { extras.append((range.upperBound, Entry(kind: .note, title: "Citations", text: citations.note))) }
        if final, finish == "cap" {
            extras.append((range.upperBound, Entry(
                kind: .failure, title: "Answer cut at the token limit (\(maxNewTokens))",
                text: "The answer above stops where generation ran out of tokens. Raise Max tokens or ask for something shorter.")))
        }

        var record = MetricsLog.agent("message", model: modelName)
        record.agentSession = sessionID
        record.agentTurn = turnIndex
        record.agentStep = step
        record.agentEffort = effort.rawValue
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
        record.citationsTotal = citations?.total
        record.citationsSeen = citations?.seenCount
        record.citationMeanOverlap = citations?.meanOverlap
        MetricsLog.save(record)
    }

    private func rebuild(_ transcript: [Transcript.Entry], live: Bool) {
        var out: [Entry] = []
        let placed = Dictionary(grouping: extras, by: \.after)
        func place(_ position: Int) { out += (placed[position] ?? []).map(\.entry) }
        place(0)
        for (index, entry) in transcript.enumerated() {
            let isLive = live && index == transcript.count - 1
            if !hidden.contains(entry.id) {
                switch entry {
                case .prompt(let prompt):
                    out.append(Entry(id: entry.id, kind: .user, title: "You", text: Self.trimmed(prompt.segments)))
                case .reasoning(let reasoning):
                    let text = Self.trimmed(reasoning.segments)
                    if !text.isEmpty { out.append(Entry(id: entry.id, kind: .thinking, title: "Thinking", text: text, live: isLive)) }
                case .response(let response):
                    let text = Self.trimmed(response.segments)
                    if !text.isEmpty { out.append(Entry(id: entry.id, kind: .answer, title: "Assistant", text: text, live: isLive)) }
                case .toolCalls(let calls):
                    for call in calls {
                        let arguments = AgentPromptBuilder.arguments(call.toolName, call.arguments)
                        out.append(Entry(
                            id: call.id, kind: .toolCall, title: call.toolName,
                            text: AgentToolCall(name: call.toolName, arguments: arguments).summary, live: isLive))
                    }
                case .toolOutput(let output):
                    out.append(Entry(
                        id: "result-" + entry.id, kind: .toolResult, title: "\(output.toolName) result", text: ChatViewModel.text(output.segments)))
                default:
                    break
                }
            }
            place(index + 1)
        }
        for index in out.indices { out[index].stats = stats[out[index].id] }
        entries = out
    }

    private static func steps(_ transcript: [Transcript.Entry], from start: Int) -> [(generated: Range<Int>, outputs: Range<Int>?)] {
        func isOutput(_ index: Int) -> Bool { if case .toolOutput = transcript[index] { true } else { false } }
        var steps: [(generated: Range<Int>, outputs: Range<Int>?)] = []
        var index = start
        while index < transcript.count {
            var end = index
            while end < transcript.count, !isOutput(end) { end += 1 }
            var after = end
            while after < transcript.count, isOutput(after) { after += 1 }
            steps.append((index..<end, end < after ? end..<after : nil))
            index = after
        }
        return steps
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

    private static func pendingCalls(_ entries: ArraySlice<Transcript.Entry>) -> [String]? {
        guard case .toolCalls(let last)? = entries.last, !(last.last?.metadata.isEmpty ?? true) else { return nil }
        return entries.flatMap { entry -> [String] in if case .toolCalls(let calls) = entry { calls.map(\.toolName) } else { [] } }
    }

    private static func lastDisplayID(_ entries: ArraySlice<Transcript.Entry>) -> String? {
        for entry in entries.reversed() {
            switch entry {
            case .toolCalls(let calls) where !calls.isEmpty: return calls.last?.id
            case .response(let response) where !trimmed(response.segments).isEmpty: return entry.id
            case .reasoning(let reasoning) where !trimmed(reasoning.segments).isEmpty: return entry.id
            default: continue
            }
        }
        return nil
    }

    private static func responseText(_ entries: ArraySlice<Transcript.Entry>) -> String {
        entries.map { if case .response(let response) = $0 { ChatViewModel.text(response.segments) } else { "" } }
            .joined().trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func trimmed(_ segments: [Transcript.Segment]) -> String {
        ChatViewModel.text(segments).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    struct CitationCheck {
        var urls: [(url: String, seen: Bool, overlap: Double)] = []
        var total: Int { urls.count }
        var seenCount: Int { urls.count { $0.seen } }
        var meanOverlap: Double {
            urls.isEmpty ? 0 : urls.map(\.overlap).reduce(0, +) / Double(urls.count)
        }
        var note: String {
            String(format: "%d/%d cited URLs were retrieved in this task · mean 8-gram overlap %.2f",
                   seenCount, total, meanOverlap)
        }
    }

    nonisolated static let citationSourceLimit = 200_000

    private func checkCitations(_ answer: String) async -> CitationCheck? {
        let (body, urls) = Self.splitSources(answer)
        guard !urls.isEmpty else { return nil }
        var lookup: [String: String] = [:]
        for (url, text) in seenPages { lookup[Self.normalized(url)] = text }
        var check = CitationCheck()
        for url in urls {
            let source = lookup[Self.normalized(url)]
            var overlap = 0.0
            if let source {
                let clipped = String(source.prefix(Self.citationSourceLimit))
                overlap = await Task.detached { Self.gramOverlap(body, clipped) }.value
            }
            check.urls.append((url, source != nil, overlap))
            AgentTools.log(String(format: "[cite] url=%@ seen=%@ overlap=%.2f", url, source == nil ? "no" : "yes", overlap))
        }
        return check
    }

    nonisolated static func splitSources(_ answer: String) -> (body: String, urls: [String]) {
        let lines = answer.split(separator: "\n", omittingEmptySubsequences: false)
        guard let start = lines.lastIndex(where: { line in
            let head = line.trimmingCharacters(in: CharacterSet(charactersIn: " \t*#-")).lowercased()
            return head.hasPrefix("sources") || head.hasPrefix("\u{51FA}\u{5178}")
                || head.hasPrefix("\u{53C2}\u{8003}")
        }) else { return (answer, []) }
        return (lines[..<start].joined(separator: "\n"), urls(in: lines[start...].joined(separator: "\n")))
    }

    nonisolated static func urls(in text: String) -> [String] {
        var found: [String] = []
        for match in text.matches(of: /https?:\/\/[^\s<>"'()\[\]]+/) {
            var url = String(match.0)
            while let last = url.last,
                  ".,;:\u{3001}\u{3002}\u{00BB}\u{201D}".contains(last) { url.removeLast() }
            if !url.isEmpty, !found.contains(url) { found.append(url) }
        }
        return found
    }

    nonisolated static func normalized(_ url: String) -> String {
        var text = url.lowercased()
        while text.hasSuffix("/") { text.removeLast() }
        return text
    }

    nonisolated static func gramOverlap(_ body: String, _ source: String, size: Int = 8) -> Double {
        func grams(_ text: String) -> [String] {
            let characters = Array(text.lowercased().filter { !$0.isWhitespace })
            guard characters.count >= size else { return [] }
            return (0...(characters.count - size)).map { String(characters[$0..<($0 + size)]) }
        }
        let needles = grams(body)
        guard !needles.isEmpty else { return 0 }
        let haystack = Set(grams(source))
        return Double(needles.count { haystack.contains($0) }) / Double(needles.count)
    }
}

extension AgentViewModel {
    static func newSessionID() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date())
    }

    static func transcriptsDirectory() -> URL {
        ModelStorage.root.appending(path: "agent-transcripts", directoryHint: .isDirectory)
    }

    func transcriptMarkdown() -> String {
        func fence(_ text: String) -> String { "````\n\(text)\n````" }
        var lines = [
            "# Agent transcript",
            "",
            "- model: \(modelName.isEmpty ? "(none)" : modelName)",
            "- session: \(sessionID ?? "(none)")",
            "- effort: \(effort.rawValue)",
            "- max steps: \(maxIterations)",
            "- page budget: \(pageCharacterBudget)",
            "- search: \(searchProvider)",
            "- date: \(currentDate ?? "(none)")",
            "- app build: \(Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?")",
            "",
        ]
        for entry in entries {
            switch entry.kind {
            case .user:
                lines += ["## You", "", entry.text, ""]
            case .thinking:
                lines += ["### Thinking (\(entry.text.count) chars)", "", fence(entry.text), ""]
            case .toolCall:
                lines += ["### Calling \(entry.title)", "", fence(entry.text), ""]
            case .toolResult:
                lines += ["### \(entry.title) (\(entry.text.count) chars)", "", fence(entry.text), ""]
            case .answer:
                lines += ["## Assistant", "", entry.text, ""]
            case .failure:
                lines += ["> **\(entry.title)**: \(entry.text)", ""]
            case .note:
                lines += ["_\(entry.title): \(entry.text)_", ""]
            }
            if let stats = entry.stats { lines += ["_\(stats)_", ""] }
        }
        return lines.joined(separator: "\n")
    }

    func copyTranscript() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(transcriptMarkdown(), forType: .string)
    }

    func saveTranscript() {
        guard let sessionID, !entries.isEmpty else { return }
        let directory = Self.transcriptsDirectory()
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data(transcriptMarkdown().utf8)
                .write(to: directory.appending(path: "\(sessionID).md"), options: .atomic)
        } catch {
            AgentTools.log("[agent] transcript save failed: \(error)")
        }
    }
}
