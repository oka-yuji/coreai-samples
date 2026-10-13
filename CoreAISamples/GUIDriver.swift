import AppKit
import ConversationFeature
import ExtractFeature
import FeatureStore
import Foundation
import GLiNERCore
import ScreenCaptureKit

@MainActor
final class GUIDriver {
    static let shared = GUIDriver()

    struct StepResult: Encodable {
        let step: Int
        let title: String
        let pass: Bool
        let detail: String
        let screenshot: String?
        let seconds: Double
    }

    static let agentTask = "Search for what Apple's MLX framework is and summarise it in three lines."
    static let stopTask = "Read https://developer.apple.com/documentation/createml and list three key points."

    private var attached = false
    private var results: [StepResult] = []
    private var lines: [String] = []
    private var sessions: [String] = []
    private var outDir = URL.temporaryDirectory
    private var maxStall = 0.0
    private var extract: Store<ExtractFeature>?

    func start(
        navigator: DemoNavigator, chat: ChatViewModel, host: ConversationHost, models: ModelsViewModel,
        extract: Store<ExtractFeature>
    ) async {
        guard let path = LaunchArguments.value("--gui-e2e"), !attached else { return }
        attached = true
        self.extract = extract
        outDir = URL(filePath: path, directoryHint: .isDirectory)
        await run(navigator: navigator, chat: chat, host: host, models: models)
    }

    private func note(_ text: String) {
        let line = "[gui] " + text
        lines.append(line)
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    private func settle() async { try? await Task.sleep(for: .milliseconds(400)) }

    private func window() -> NSWindow? { NSApp.windows.first { $0.isVisible } }

    private func wait(seconds: Double, until condition: () -> Bool) async -> Bool {
        let deadline = Date().addingTimeInterval(seconds)
        while !condition() {
            if Date() > deadline { return false }
            try? await Task.sleep(for: .milliseconds(100))
        }
        return true
    }

    private static func head(_ text: String, _ limit: Int = 300) -> String {
        text.count <= limit ? text : String(text.prefix(limit)) + "…"
    }

    private static func uniqueColors(_ png: Data) -> Int {
        guard let rep = NSBitmapImageRep(data: png) else { return -1 }
        var seen = Set<UInt32>()
        for y in stride(from: 0, to: rep.pixelsHigh, by: 8) {
            for x in stride(from: 0, to: rep.pixelsWide, by: 8) {
                guard let color = rep.colorAt(x: x, y: y) else { continue }
                seen.insert(
                    UInt32(color.redComponent * 255) << 16 | UInt32(color.greenComponent * 255) << 8
                        | UInt32(color.blueComponent * 255))
            }
        }
        return seen.count
    }

    private func captureWithScreenCaptureKit(_ window: NSWindow) async -> Data? {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            let id = CGWindowID(window.windowNumber)
            guard let target = content.windows.first(where: { $0.windowID == id }) else {
                note("capture: window \(id) is not in the shareable list")
                return nil
            }
            let configuration = SCStreamConfiguration()
            let scale = window.backingScaleFactor
            configuration.width = Int(target.frame.width * scale)
            configuration.height = Int(target.frame.height * scale)
            configuration.showsCursor = false
            let image = try await SCScreenshotManager.captureImage(
                contentFilter: SCContentFilter(desktopIndependentWindow: target), configuration: configuration)
            return NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])
        } catch {
            note("capture: ScreenCaptureKit failed: \(error)")
            return nil
        }
    }

    private static func captureWithCacheDisplay(_ window: NSWindow) -> Data? {
        guard let view = window.contentView, let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else {
            return nil
        }
        view.cacheDisplay(in: view.bounds, to: rep)
        return rep.representation(using: .png, properties: [:])
    }

    private func shoot(_ index: Int, _ slug: String) async -> (name: String?, detail: String) {
        guard let window = window() else { return (nil, "no visible window") }
        var method = "screenCaptureKit"
        var data = await captureWithScreenCaptureKit(window)
        if data == nil {
            method = "cacheDisplay"
            data = Self.captureWithCacheDisplay(window)
        }
        guard let data else { return (nil, "capture failed") }
        let name = String(format: "%02d-%@.png", index, slug)
        do {
            try data.write(to: outDir.appending(path: name), options: .atomic)
        } catch {
            return (nil, "write failed: \(error)")
        }
        let detail = "\(method) \(data.count)B colors=\(Self.uniqueColors(data))"
        note("shot \(name) \(detail)")
        return (name, detail)
    }

    private func step(_ index: Int, _ slug: String, _ title: String, _ body: () async -> (Bool, String)) async {
        note("step \(index) \(slug) start")
        let started = Date()
        let (pass, detail) = await body()
        await settle()
        let shot = await shoot(index, slug)
        let seconds = Date().timeIntervalSince(started)
        results.append(
            StepResult(
                step: index, title: title, pass: pass, detail: detail + " | shot: " + shot.detail,
                screenshot: shot.name, seconds: (seconds * 100).rounded() / 100))
        note(
            String(
                format: "step %d %@ %@ %.1fs %@", index, slug, pass ? "PASS" : "FAIL", seconds, Self.head(detail, 600)))
    }

    private func ask(_ host: ConversationHost, _ text: String, timeout: Double = 900) async -> Bool {
        host.chat.input = text
        guard host.submit() else {
            note(
                "send refused: phase=\(host.chat.phase) running=\(host.state.isRunning) contextFull=\(host.contextFull)"
            )
            return false
        }
        return await wait(seconds: timeout) { !host.state.isRunning && !host.chat.isResponding }
    }

    private static func turnText(_ host: ConversationHost) -> (thinking: Int, answer: Int) {
        let turn = host.state.entries.reversed().prefix { $0.kind != .user }
        let thinking = turn.filter { $0.kind == .thinking }.map(\.text.count).reduce(0, +)
        let answer = turn.filter { $0.kind == .answer || ($0.kind == .failure && $0.title == "Stopped") }
            .map(\.text.count).reduce(0, +)
        return (thinking, answer)
    }

    private func run(navigator: DemoNavigator, chat: ChatViewModel, host: ConversationHost, models: ModelsViewModel)
        async
    {
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let metricsBefore =
            (try? String(contentsOf: MetricsLog.fileURL, encoding: .utf8))?.split(separator: "\n").count ?? 0
        note("out=\(outDir.path(percentEncoded: false)) contextLimit=\(chat.contextLimit)")
        if LaunchArguments.value("--extract") != nil, let extract {
            await runExtract(navigator: navigator, store: extract)
            await finish(metricsBefore: metricsBefore)
            return
        }
        guard let path = LaunchArguments.value("--model") else {
            note("--model <folder> is required")
            await finish(metricsBefore: metricsBefore)
            return
        }
        let url = URL(filePath: path, directoryHint: .isDirectory)
        let preferences = host.preferences
        preferences.persists = false
        preferences.effort = LaunchArguments.value("--effort").flatMap(AgentReasoningEffort.init(argument:)) ?? .medium
        if let tokens = LaunchArguments.value("--max-tokens").flatMap(Int.init) { preferences.maxTokens = tokens }
        NSApp.activate()
        _ = await wait(seconds: 30) { self.window() != nil }
        await settle()

        await step(1, "models", "Models lists the opened folder") {
            navigator.selection = .models
            models.open(url)
            var cleared = "no"
            if LaunchArguments.flag("--clear-prepared") {
                models.clearPrepared(url)
                cleared = "yes"
            }
            await self.settle()
            let folder = models.folders.first { $0.id == ModelStorage.key(url) }
            return (
                folder?.name != nil,
                "folder=\(folder?.id ?? "none") name=\(folder?.name ?? "none") clearPrepared=\(cleared) message=\(models.message ?? "")"
            )
        }

        var preparingShown = false
        let heartbeat = Task {
            var last = ContinuousClock.now
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(20))
                let now = ContinuousClock.now
                self.maxStall = max(self.maxStall, (now - last) / .seconds(1))
                last = now
            }
        }

        await step(2, "loading", "Chat shows the loading state while the model opens") {
            navigator.selection = .chat
            Task { await chat.load(url) }
            _ = await self.wait(seconds: 5) { chat.isLoading }
            if case .loading(true) = chat.phase { preparingShown = true }
            try? await Task.sleep(for: .seconds(2.5))
            return (chat.isLoading || chat.phase == .ready, "phase=\(chat.phase) preparingShown=\(preparingShown)")
        }

        await step(3, "ready", "The model is open, the window stayed responsive, and the chat is ready") {
            let loaded = await self.wait(seconds: 900) { !chat.isLoading }
            heartbeat.cancel()
            let stats = chat.loadStats
            let responsive = self.maxStall < 0.5
            let honest = stats.map { $0.cached || preparingShown } ?? false
            return (
                loaded && chat.phase == .ready && responsive && honest,
                String(
                    format:
                        "phase=%@ cached=%@ specializeSeconds=%.2f loadSeconds=%.2f maxMainStall=%.3f preparingShown=%@ note=%@",
                    "\(chat.phase)", stats.map { String($0.cached) } ?? "nil", stats?.specializeSeconds ?? -1,
                    chat.loadSeconds,
                    self.maxStall, String(preparingShown), chat.loadNote)
            )
        }

        if LaunchArguments.flag("--agent") {
            await runAgent(navigator: navigator, host: host)
            await finish(metricsBefore: metricsBefore)
            return
        }

        if LaunchArguments.flag("--conversation") {
            await runConversation(host: host)
            await finish(metricsBefore: metricsBefore)
            return
        }

        if let question = LaunchArguments.value("--rag") {
            await runRAG(host: host, question: question)
            await finish(metricsBefore: metricsBefore)
            return
        }

        await step(4, "chat", "One turn streams to the end and the status line shows the numbers") {
            let done = await self.ask(host, "Explain in two sentences what a context window is in a language model.")
            let last = host.runner.session.lastGeneration
            let pass =
                done && last?.error == nil && (last?.outputTokens ?? 0) > 0
                && ["stopToken", "maxTokens"].contains(last?.stopReason ?? "")
            return (
                pass,
                "status=\(last?.statusLine ?? "none") thoughtChars=\(Self.turnText(host).thinking) answer=\(Self.head(last?.answer ?? ""))"
            )
        }

        await step(5, "generating", "A long reply is streaming") {
            chat.input =
                "Write a detailed essay of at least 1500 words about the history of computing, from the abacus to modern AI. Use multiple sections with headings."
            guard host.submit() else { return (false, "send refused") }
            let grew = await self.wait(seconds: 300) {
                guard host.state.turns.count == 2 else { return false }
                let text = Self.turnText(host)
                return text.thinking + text.answer > 300
            }
            let text = Self.turnText(host)
            return (
                grew && host.state.isRunning,
                "running=\(host.state.isRunning) thoughtChars=\(text.thinking) answerChars=\(text.answer)"
            )
        }

        await step(6, "stopped", "Stop ends the reply within three seconds and keeps what was written") {
            let started = ContinuousClock.now
            host.stop()
            let halted = await self.wait(seconds: 3) { !host.state.isRunning }
            let seconds = (ContinuousClock.now - started) / .seconds(1)
            let text = Self.turnText(host)
            let last = host.runner.session.lastGeneration
            return (
                halted && last?.stopReason == "cancelled" && text.thinking + text.answer > 0,
                String(
                    format: "halted=%@ seconds=%.3f keptChars=%d status=%@", String(halted), seconds,
                    text.thinking + text.answer, last?.statusLine ?? "none")
            )
        }

        await step(7, "new-chat", "New starts an empty conversation") {
            host.startNew()
            await self.settle()
            return (
                host.state.entries.isEmpty && host.runner.session.lastGeneration == nil && chat.phase == .ready,
                "entries=\(host.state.entries.count) phase=\(chat.phase) status=\(host.state.statusLine)"
            )
        }

        if LaunchArguments.value("--context-limit") != nil {
            await step(8, "context-full", "A long chat with a small context limit ends with a notice, not an error") {
                var sent = 0
                while !host.contextFull, sent < 40 {
                    sent += 1
                    guard
                        await self.ask(host, "Tell me one more fact about the history of computing, in about 80 words.")
                    else { break }
                    self.note(
                        "context turn \(sent): \(host.runner.session.lastGeneration?.statusLine ?? "none") "
                            + "notice=\(host.state.turns.last?.failure ?? "none")")
                }
                let last = host.runner.session.lastGeneration
                return (
                    host.contextFull && chat.phase == .ready && last?.error == nil,
                    "turns=\(sent) contextFull=\(host.contextFull) notice=\(host.state.turns.last?.failure ?? "none") stop=\(last?.stopReason ?? "none")"
                )
            }
        }

        await finish(metricsBefore: metricsBefore)
    }

    private func runConversation(host: ConversationHost) async {
        host.preferences.notesFolderPath = outDir.appending(path: "notes").path(percentEncoded: false)
        var previous: StepStats?
        let strictReuse = host.chat.modelName.hasPrefix("qwen3_8")
        func turn(_ mode: ConversationMode, _ text: String) async -> (done: Bool, reused: Bool, reuse: String) {
            host.store.send(.modeChanged(mode))
            let sent = host.state.turns.count
            let divergedBefore = host.runner.session.divergeCount
            let done = await ask(host, text)
            let turns = host.state.turns
            let steps = sent < turns.count ? turns[sent].steps.sorted { $0.step < $1.step } : []
            let diverged = host.runner.session.divergeCount - divergedBefore
            defer { previous = turns.last?.steps.max { $0.step < $1.step } ?? previous }
            guard let first = steps.first else { return (done, false, "steps=0") }
            let repairs = turns.count - sent - 1
            guard let previous else {
                return (done, true, "reused=\(first.reusedTokens) steps=\(steps.count) repairs=\(repairs)")
            }
            let expected = previous.promptTokens + previous.generatedTokens
            let kept = first.reusedTokens >= expected - 1
            return (
                done, !strictReuse || (kept && diverged == 0),
                "reused=\(first.reusedTokens) previousPromptPlusOutput=\(expected) gap=\(expected - first.reusedTokens) "
                    + "steps=\(steps.count) repairs=\(repairs) diverges=\(diverged) strictReuse=\(strictReuse)"
            )
        }
        func lookups() -> Int {
            host.state.entries.count { $0.kind == .toolCall && ["web_search", "fetch_page"].contains($0.title) }
        }
        func ended(_ mode: ConversationMode) -> Bool {
            guard let last = host.state.turns.last else { return false }
            return last.mode == mode && !host.state.isRunning && last.failure == nil
        }

        await step(4, "conversation-chat", "Chat: a question is answered in the conversation") {
            let result = await turn(.chat, "In one sentence, what is a context window in a language model?")
            return (result.done && result.reused && ended(.chat), result.reuse + " status=\(host.state.statusLine)")
        }

        await step(5, "conversation-work", "Work: research and a note continue the same conversation") {
            let before = lookups()
            let result = await turn(
                .work, "Search the web for what Apple's MLX framework is and save a two-line note titled MLX summary.")
            let calls = host.state.entries.filter { $0.kind == .toolCall }.map(\.title)
            let notes = host.runner.tools.notesWritten.map(\.key)
            return (
                result.done && result.reused && ended(.work) && lookups() > before && !notes.isEmpty,
                result.reuse + " calls=\(calls) notes=\(notes) guard=\(host.runner.citationGuard.outcome ?? "none")"
            )
        }

        await step(6, "conversation-chat-again", "Chat: a follow-up question continues the same timeline") {
            let result = await turn(.chat, "Thanks. In one sentence, what did you just look up?")
            let prompts = host.state.entries.filter { $0.kind == .user }.count
            return (
                result.done && result.reused && ended(.chat) && prompts >= 3,
                result.reuse + " userEntries=\(prompts) turns=\(host.state.turns.map(\.mode.rawValue))"
            )
        }

        await step(7, "conversation-save-refused", "Chat: saving a note is refused and Continue in Work is offered") {
            host.startNew()
            previous = nil
            let result = await turn(
                .chat, "Use write_note to save a note titled Shopping list with the content: eggs, milk.")
            let refused = host.state.entries.contains {
                $0.kind == .toolResult && $0.text.hasPrefix(TurnPolicy.savingRefusal)
            }
            return (
                result.done && host.state.refusedSave && refused,
                "refusedSave=\(host.state.refusedSave) refusalShown=\(refused) " + result.reuse
            )
        }

        await step(8, "conversation-continue", "Continue in Work saves the note in the same conversation") {
            host.continueInWork()
            let done = await wait(seconds: 900) { !host.state.isRunning }
            let notes = host.runner.tools.notesWritten.map(\.key)
            return (
                done && host.state.turns.last?.mode == .work && !notes.isEmpty,
                "mode=\(host.state.turns.last?.mode.rawValue ?? "none") notes=\(notes) guard=\(host.runner.citationGuard.outcome ?? "none")"
            )
        }

        await step(
            9, "conversation-running",
            "While a turn runs the mode and Send are locked, and Stop ends it within three seconds"
        ) {
            host.store.send(.modeChanged(.work))
            host.chat.input = Self.stopTask
            guard host.submit() else { return (false, "send refused") }
            let grew = await wait(seconds: 180) { (host.state.entries.last?.text.count ?? 0) > 20 }
            host.chat.input = "Another question"
            host.store.send(.modeChanged(.chat))
            let locked = host.state.isRunning && !host.canSend && host.state.mode == .work
            let started = ContinuousClock.now
            host.stop()
            let halted = await wait(seconds: 3) { !host.state.isRunning }
            let seconds = (ContinuousClock.now - started) / .seconds(1)
            host.chat.input = ""
            let stopped = host.state.entries.last { $0.kind == .failure && $0.title == "Stopped" }
            return (
                grew && locked && halted && stopped != nil,
                String(
                    format: "grew=%@ locked=%@ halted=%@ seconds=%.3f stopped=%@", String(grew), String(locked),
                    String(halted), seconds, Self.head(stopped?.text ?? "none", 200))
            )
        }
    }

    private func runAgent(navigator: DemoNavigator, host: ConversationHost) async {
        let preferences = host.preferences
        preferences.effort = .low
        let policy = TurnPolicy(.work, settings: preferences.settings)
        note(
            "agent effort=\(preferences.effort.rawValue) steps=\(policy.maxSteps) tokens=\(policy.maxTokens) "
                + "budget=\(preferences.pageBudget) search=\(preferences.searchProvider) fileOps=\(preferences.fileOperations)"
        )

        await step(4, "agent-task", "Work: the task is sent and the first step streams") {
            navigator.selection = .chat
            await self.settle()
            host.store.send(.modeChanged(.work))
            host.chat.input = Self.agentTask
            guard host.submit() else { return (false, "send refused") }
            self.sessions.append(host.state.id)
            let streaming = await self.wait(seconds: 120) { host.state.entries.contains { $0.kind != .user } }
            return (
                streaming && host.state.isRunning,
                "running=\(host.state.isRunning) kinds=\(host.state.entries.map { "\($0.kind)" }) status=\(host.state.statusLine)"
            )
        }

        await step(5, "agent-tools", "Work: tool calls and their results are listed") {
            let listed = await self.wait(seconds: 600) {
                host.state.entries.contains { $0.kind == .toolCall }
                    && host.state.entries.contains { $0.kind == .toolResult }
            }
            let entries = host.state.entries
            let calls = entries.filter { $0.kind == .toolCall }.map { "\($0.title) {\(Self.head($0.text, 120))}" }
            let results = entries.filter { $0.kind == .toolResult }.map { "\($0.title) \($0.text.count) chars" }
            return (listed, "calls=\(calls) results=\(results) status=\(host.state.statusLine)")
        }

        await step(6, "agent-answer", "Work: the answer ends with Sources and every step has its numbers") {
            let done = await self.wait(seconds: 900) { !host.state.isRunning }
            let answer = host.state.entries.last { $0.kind == .answer }?.text ?? ""
            let stats = host.state.entries.compactMap(\.stats)
            let finish = host.runner.session.lastFinish
            return (
                done && answer.contains("Sources") && finish == "eos" && !stats.isEmpty,
                "finished=\(done) sources=\(answer.contains("Sources")) finish=\(finish ?? "none") "
                    + "stats=\(stats.joined(separator: " ;; ")) answer=\(Self.head(answer, 800))"
            )
        }

        await step(7, "agent-stop", "Work: Stop halts a running task within three seconds and keeps what was written") {
            let mark = host.state.entries.count
            host.chat.input = Self.stopTask
            guard host.submit() else { return (false, "send refused") }
            let grew = await self.wait(seconds: 180) {
                host.state.entries.count > mark + 1 && (host.state.entries.last?.text.count ?? 0) > 20
            }
            let started = ContinuousClock.now
            host.stop()
            let halted = await self.wait(seconds: 3) { !host.state.isRunning }
            let seconds = (ContinuousClock.now - started) / .seconds(1)
            _ = await self.wait(seconds: 120) { !host.state.isRunning }
            let stopped = host.state.entries.last { $0.kind == .failure && $0.title == "Stopped" }
            return (
                grew && halted && stopped != nil,
                String(
                    format: "grew=%@ halted=%@ seconds=%.3f stopped=%@ status=%@", String(grew), String(halted),
                    seconds,
                    Self.head(stopped?.text ?? "none", 200), host.state.statusLine)
            )
        }

        await step(8, "agent-new-task", "Work: New clears the conversation") {
            let previous = host.state.id
            host.startNew()
            await self.settle()
            return (
                host.state.entries.isEmpty && !host.state.isRunning && host.state.mode == .work,
                "entries=\(host.state.entries.count) previous=\(previous) mode=\(host.state.mode) status=\(host.state.statusLine)"
            )
        }
    }

    private func runRAG(host: ConversationHost, question: String) async {
        let chat = host.chat
        if let window = window(), let visible = window.screen?.visibleFrame {
            let height = min(1_300, visible.height)
            window.setFrame(
                NSRect(x: visible.minX + 40, y: visible.maxY - height, width: 1_200, height: height), display: true)
        }

        await step(4, "rag-switches", "RAG: the RAG, Web and Docs switches are on in the status bar") {
            chat.ragEnabled = true
            chat.ragWeb = !SelfTest.isOff(LaunchArguments.value("--rag-web"))
            chat.ragLocal = !SelfTest.isOff(LaunchArguments.value("--rag-local"))
            await self.settle()
            return (
                chat.ragIsOn,
                "rag=\(chat.ragEnabled) web=\(chat.ragWeb) docs=\(chat.ragLocal) window=\(self.window()?.frame.size ?? .zero)"
            )
        }

        await step(
            5, "rag-answer", "RAG: the answer, its numbered sources and the status line, with no turn added to the chat"
        ) {
            let done = await self.ask(host, question)
            let turn = chat.ragTurns.last
            let sources = turn?.sources ?? []
            let pass =
                done && chat.last?.error == nil && !sources.isEmpty && !(turn?.response ?? "").isEmpty
                && host.state.entries.isEmpty
                && chat.statusLine.contains("prompt(rag)")
            return (
                pass,
                "status=\(chat.statusLine) chatEntries=\(host.state.entries.count) sources=\(sources.count) notes=\(turn?.notes ?? []) "
                    + "first=\(sources.first ?? "none") answer=\(Self.head(turn?.response ?? "", 600))"
            )
        }
    }

    private func runExtract(navigator: DemoNavigator, store: Store<ExtractFeature>) async {
        let examples = ExtractResources.examples
        guard let asset = LaunchArguments.value("--extract"), let example = examples.notes.first else {
            note("--extract needs an asset folder and the bundled examples")
            return
        }
        NSApp.activate()
        _ = await wait(seconds: 30) { self.window() != nil }
        if let window = window(), let visible = window.screen?.visibleFrame {
            let height = min(900, visible.height)
            window.setFrame(
                NSRect(x: visible.minX + 40, y: visible.maxY - height, width: 1_320, height: height), display: true)
        }
        func extracted() async -> Bool {
            _ = await wait(seconds: 120) {
                switch store.state.phase {
                case .ready, .failed: true
                case .idle, .extracting: false
                }
            }
            return store.state.phase == .ready
        }
        func detected() -> [String] {
            (store.state.result?.mapping.fields ?? []).filter { !$0.candidates.isEmpty }.map(\.field.key)
        }

        await step(1, "extract-initial", "Extract opens in the label mode and waits for a note") {
            navigator.selection = .extract
            store.send(.assetFolderChosen(path: asset))
            _ = await self.wait(seconds: 900) {
                switch store.state.asset {
                case .ready, .failed: true
                case .none, .opening: false
                }
            }
            let opened = if case .ready = store.state.asset { true } else { false }
            return (
                opened && store.state.text.isEmpty && store.state.mode == .labels,
                "asset=\(store.state.asset) mode=\(store.state.mode) labels=\(store.state.labels.fields.map(\.key))"
            )
        }

        await step(2, "extract-labels", "A pasted note gets the spans of the labels, colored in the note by label") {
            store.send(.textChanged(example))
            let done = await extracted()
            let timing = store.state.result.map { ExtractRunner.describe($0.timing) } ?? "none"
            let spans = (store.state.result?.mapping.fields ?? []).map { field in
                "\(field.field.key)=\(field.candidates.map { "\($0.span.text) \($0.span.probability)" })"
            }
            return (
                done && !detected().isEmpty && store.state.mode == .labels,
                "phase=\(store.state.phase) spans=\(spans) timing=\(timing)"
            )
        }

        await step(3, "extract-experiment", "The Experiment switch maps the note onto the form, with suggestions") {
            store.send(.modeChanged(.form))
            let done = await extracted()
            let suggested = (store.state.result?.mapping.fields ?? []).contains { field in
                field.candidates.contains { !$0.suggestions.isEmpty }
            }
            let cues = store.state.result?.mapping.corrections.map(\.word) ?? []
            return (
                done && store.state.mode == .form && suggested && !detected().isEmpty,
                "phase=\(store.state.phase) detected=\(detected()) cues=\(cues) "
                    + "filter=\(store.state.candidateFilter.name)"
            )
        }

        await step(4, "extract-field-added", "An added field gets candidates and its evidence is emphasized") {
            let field = examples.extraField
            store.send(.fieldsChanged(FormSchema(fields: store.state.form.fields + [field])))
            store.send(.select(field: field.key))
            let done = await extracted()
            let added = store.state.result?.mapping.fields.first { $0.field.key == field.key }
            let texts = added?.candidates.map(\.span.text) ?? []
            return (
                done && !texts.isEmpty && store.state.selectedField == field.key,
                "phase=\(store.state.phase) added=\(field.key) candidates=\(texts) detected=\(detected())"
            )
        }

        await step(
            5, "extract-confirmed",
            "Two confirmed fields keep their choices, and the JSON keeps each section, the filter and the versions"
        ) {
            let fields = store.state.result?.mapping.fields ?? []
            var confirmed: [String: ExtractChoice] = [:]
            for field in fields where confirmed.count < 2 {
                guard let first = field.candidates.first, let last = field.candidates.last else { continue }
                switch field.field.kind {
                case .number where !confirmed.values.contains { if case .value = $0 { true } else { false } }:

                    confirmed[field.field.key] = .value(text: last.span.text)
                case .list where !confirmed.values.contains { if case .option = $0 { true } else { false } }:
                    guard let option = first.suggestions.first?.option else { continue }
                    confirmed[field.field.key] = .option(name: option)
                default:
                    continue
                }
            }
            for (key, choice) in confirmed { store.send(.confirm(field: key, choice: choice)) }
            store.send(.exportRequested)
            _ = await self.wait(seconds: 10) { store.state.export != .none }
            guard case .saved(let path) = store.state.export, let data = FileManager.default.contents(atPath: path),
                let document = try? JSONDecoder().decode(ExportDocument.self, from: data)
            else { return (false, "export=\(store.state.export)") }
            try? data.write(to: self.outDir.appending(path: "export.json"), options: .atomic)
            let exported = Dictionary(uniqueKeysWithValues: document.confirmations.map { ($0.key, $0.choice) })
            let model = document.model
            let pass =
                confirmed.count == 2 && store.state.confirmations == confirmed && exported == confirmed
                && document.input.text == example && model?.fields.count == document.form.fields.count
                && document.rules != nil && document.mode == "form"
                && model?.candidateFilter == "clauseFieldNameAndEquipmentNumber"
                && model?.assetRevision != nil && model?.appVersion != nil
            return (
                pass,
                "confirmed=\(confirmed) export=\(path) mode=\(document.mode) modelFields=\(model?.fields.count ?? 0) "
                    + "filter=\(model?.candidateFilter ?? "none") revision=\(model?.assetRevision ?? "none") "
                    + "app=\(model?.appVersion ?? "none") cues=\(document.rules?.correctionCues.map(\.word) ?? [])"
            )
        }
    }

    private func finish(metricsBefore: Int) async {
        for id in sessions {
            let source = ConversationRunner.transcriptsDirectory().appending(path: "\(id).md")
            do {
                try FileManager.default.copyItem(at: source, to: outDir.appending(path: "transcript-\(id).md"))
            } catch {
                note("transcript missing for \(id): \(error)")
            }
        }
        await MetricsLog.flush()
        let metrics = ((try? String(contentsOf: MetricsLog.fileURL, encoding: .utf8)) ?? "").split(separator: "\n")
            .dropFirst(metricsBefore)
        try? Data((metrics.joined(separator: "\n") + "\n").utf8).write(
            to: outDir.appending(path: "metrics.jsonl"), options: .atomic)
        note("metrics rows written by this run: \(metrics.count)")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try? encoder.encode(results).write(to: outDir.appending(path: "results.json"), options: .atomic)
        let passed = results.count { $0.pass }
        let pass = !results.isEmpty && passed == results.count
        note("GUI E2E \(pass ? "PASS" : "FAIL") \(passed)/\(results.count)")
        try? Data((lines.joined(separator: "\n") + "\n").utf8).write(
            to: outDir.appending(path: "driver.log"), options: .atomic)
        exit(pass ? 0 : 1)
    }
}
