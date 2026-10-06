import AppKit
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

    func attach(
        navigator: DemoNavigator, chat: ChatViewModel, agent: AgentViewModel, models: ModelsViewModel,
        extract: Store<ExtractFeature>
    ) {
        guard let path = LaunchArguments.value("--gui-e2e"), !attached else { return }
        attached = true
        self.extract = extract
        outDir = URL(filePath: path, directoryHint: .isDirectory)
        Task { await run(navigator: navigator, chat: chat, agent: agent, models: models) }
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

    private func ask(_ chat: ChatViewModel, _ text: String, timeout: Double = 900) async -> Bool {
        chat.input = text
        guard chat.send() else {
            note("send refused: phase=\(chat.phase) responding=\(chat.isResponding) contextFull=\(chat.contextFull)")
            return false
        }
        return await wait(seconds: timeout) { !chat.isResponding }
    }

    private func run(navigator: DemoNavigator, chat: ChatViewModel, agent: AgentViewModel, models: ModelsViewModel)
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
        if let tokens = LaunchArguments.value("--max-tokens").flatMap(Int.init) { chat.maxTokens = tokens }
        if let effort = LaunchArguments.value("--effort").flatMap({ ChatViewModel.Reasoning(rawValue: $0.capitalized) })
        {
            chat.reasoning = effort
        }
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
            await runAgent(navigator: navigator, chat: chat, agent: agent)
            await finish(metricsBefore: metricsBefore)
            return
        }

        if let question = LaunchArguments.value("--rag") {
            await runRAG(chat: chat, question: question)
            await finish(metricsBefore: metricsBefore)
            return
        }

        await step(4, "chat", "One turn streams to the end and the status line shows the numbers") {
            let done = await self.ask(chat, "Explain in two sentences what a context window is in a language model.")
            let last = chat.last
            let pass =
                done && last?.error == nil && (last?.outputTokens ?? 0) > 0
                && ["stopToken", "maxTokens"].contains(last?.stopReason ?? "")
            return (
                pass,
                "status=\(chat.statusLine) thoughtChars=\(chat.turns.last?.reasoning.count ?? 0) answer=\(Self.head(last?.answer ?? ""))"
            )
        }

        await step(5, "generating", "A long reply is streaming") {
            chat.input =
                "Write a detailed essay of at least 1500 words about the history of computing, from the abacus to modern AI. Use multiple sections with headings."
            guard chat.send() else { return (false, "send refused") }
            let grew = await self.wait(seconds: 300) {
                guard let turn = chat.turns.last, chat.turns.count == 2 else { return false }
                return turn.reasoning.count + turn.response.count > 300
            }
            let turn = chat.turns.last
            return (
                grew && chat.isResponding,
                "responding=\(chat.isResponding) thoughtChars=\(turn?.reasoning.count ?? 0) answerChars=\(turn?.response.count ?? 0)"
            )
        }

        await step(6, "stopped", "Stop ends the reply within three seconds and keeps what was written") {
            let started = ContinuousClock.now
            chat.stop()
            let halted = await self.wait(seconds: 3) { !chat.isResponding }
            let seconds = (ContinuousClock.now - started) / .seconds(1)
            let turn = chat.turns.last
            let kept = (turn?.reasoning.count ?? 0) + (turn?.response.count ?? 0)
            return (
                halted && chat.last?.stopReason == "cancelled" && kept > 0,
                String(
                    format: "halted=%@ seconds=%.3f keptChars=%d status=%@", String(halted), seconds, kept,
                    chat.statusLine)
            )
        }

        await step(7, "new-chat", "New Chat starts an empty conversation") {
            chat.newChat()
            await self.settle()
            return (
                chat.entries.isEmpty && chat.last == nil && chat.phase == .ready,
                "entries=\(chat.entries.count) phase=\(chat.phase) status=\(chat.statusLine)"
            )
        }

        if LaunchArguments.value("--context-limit") != nil {
            await step(8, "context-full", "A long chat with a small context limit ends with a notice, not an error") {
                var sent = 0
                while !chat.contextFull, sent < 40 {
                    sent += 1
                    guard
                        await self.ask(chat, "Tell me one more fact about the history of computing, in about 80 words.")
                    else { break }
                    self.note("context turn \(sent): \(chat.statusLine) notice=\(chat.contextNotice ?? "none")")
                }
                return (
                    chat.contextFull && chat.contextNotice != nil && chat.phase == .ready && chat.last?.error == nil,
                    "turns=\(sent) contextFull=\(chat.contextFull) notice=\(chat.contextNotice ?? "none") stop=\(chat.last?.stopReason ?? "none")"
                )
            }
        }

        await finish(metricsBefore: metricsBefore)
    }

    private func runAgent(navigator: DemoNavigator, chat: ChatViewModel, agent: AgentViewModel) async {
        agent.persistsSettings = false
        agent.effort = .low
        note(
            "agent effort=\(agent.effort.rawValue) steps=\(agent.maxIterations) tokens=\(agent.maxNewTokens) "
                + "budget=\(agent.pageCharacterBudget) search=\(agent.searchProvider) fileOps=\(agent.allowFileOperations)"
        )

        await step(4, "agent-task", "Agent: the task is sent and the first step streams") {
            navigator.selection = .agent
            await self.settle()
            agent.input = Self.agentTask
            guard agent.send(chat: chat) else { return (false, "send refused") }
            if let id = agent.sessionID { self.sessions.append(id) }
            let streaming = await self.wait(seconds: 120) { agent.entries.contains { $0.kind != .user } }
            return (
                streaming && agent.running,
                "running=\(agent.running) kinds=\(agent.entries.map { "\($0.kind)" }) status=\(agent.statusLine)"
            )
        }

        await step(5, "agent-tools", "Agent: tool calls and their results are listed") {
            let listed = await self.wait(seconds: 600) {
                agent.entries.contains { $0.kind == .toolCall } && agent.entries.contains { $0.kind == .toolResult }
            }
            let calls = agent.entries.filter { $0.kind == .toolCall }.map { "\($0.title) {\(Self.head($0.text, 120))}" }
            let results = agent.entries.filter { $0.kind == .toolResult }.map { "\($0.title) \($0.text.count) chars" }
            return (listed, "calls=\(calls) results=\(results) status=\(agent.statusLine)")
        }

        await step(6, "agent-answer", "Agent: the answer ends with Sources and every step has its numbers") {
            let done = await self.wait(seconds: 900) { !agent.running }
            let answer = agent.entries.last { $0.kind == .answer }?.text ?? ""
            let stats = agent.entries.compactMap(\.stats)
            return (
                done && answer.contains("Sources") && agent.lastFinishReason == "eos" && !stats.isEmpty,
                "finished=\(done) sources=\(answer.contains("Sources")) finish=\(agent.lastFinishReason ?? "none") "
                    + "stats=\(stats.joined(separator: " ;; ")) answer=\(Self.head(answer, 800))"
            )
        }

        await step(7, "agent-stop", "Agent: Stop halts a running task within three seconds and keeps what was written")
        {
            let mark = agent.entries.count
            agent.input = Self.stopTask
            guard agent.send(chat: chat) else { return (false, "send refused") }
            let grew = await self.wait(seconds: 180) {
                agent.entries.count > mark + 1 && (agent.entries.last?.text.count ?? 0) > 20
            }
            let started = ContinuousClock.now
            agent.stop()
            let halted = await self.wait(seconds: 3) { !agent.running }
            let seconds = (ContinuousClock.now - started) / .seconds(1)
            _ = await self.wait(seconds: 120) { !agent.running }
            let stopped = agent.entries.last { $0.kind == .failure && $0.title == "Stopped" }
            return (
                grew && halted && stopped != nil,
                String(
                    format: "grew=%@ halted=%@ seconds=%.3f stopped=%@ status=%@", String(grew), String(halted),
                    seconds,
                    Self.head(stopped?.text ?? "none", 200), agent.statusLine)
            )
        }

        await step(8, "agent-new-task", "Agent: New Task clears the transcript") {
            agent.reset()
            await self.settle()
            return (
                agent.entries.isEmpty && agent.sessionID == nil && !agent.running,
                "entries=\(agent.entries.count) session=\(agent.sessionID ?? "none") status=\(agent.statusLine)"
            )
        }
    }

    private func runRAG(chat: ChatViewModel, question: String) async {
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
            let done = await self.ask(chat, question)
            let turn = chat.turns.last
            let sources = turn?.sources ?? []
            let pass =
                done && chat.last?.error == nil && !sources.isEmpty && !(turn?.response ?? "").isEmpty
                && chat.entries.isEmpty
                && chat.statusLine.contains("prompt(rag)")
            return (
                pass,
                "status=\(chat.statusLine) chatEntries=\(chat.entries.count) sources=\(sources.count) notes=\(turn?.notes ?? []) "
                    + "first=\(sources.first ?? "none") answer=\(Self.head(turn?.response ?? "", 600))"
            )
        }
    }

    /// `--extract <asset folder>`: open the asset, paste the example note, add a field, confirm two fields and export
    /// the JSON. The folder is the flag's value: AppKit reads arguments as `-key value` pairs and opens a path left
    /// alone as a document, and then SwiftUI opens no window.
    private func runExtract(navigator: DemoNavigator, store: Store<ExtractFeature>) async {
        guard let asset = LaunchArguments.value("--extract"), let examples = ExtractResources.examples,
            let example = examples.notes.first
        else {
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

        await step(1, "extract-initial", "Extract opens the asset folder and waits for a note") {
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
                opened && store.state.text.isEmpty,
                "asset=\(store.state.asset) fields=\(store.state.form.fields.map(\.key))"
            )
        }

        await step(2, "extract-pasted", "A pasted note gets candidates, colored in the note by field") {
            store.send(.textChanged(example))
            let done = await extracted()
            let timing = store.state.result.map { ExtractView.describe($0.timing) } ?? "none"
            let cues = store.state.result?.mapping.corrections.map(\.word) ?? []
            return (
                done && !detected().isEmpty,
                "phase=\(store.state.phase) detected=\(detected()) cues=\(cues) timing=\(timing)"
            )
        }

        await step(3, "extract-field-added", "An added field gets candidates and its evidence is emphasized") {
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
            4, "extract-confirmed", "Two confirmed fields keep their choices and the JSON keeps each section apart"
        ) {
            let fields = store.state.result?.mapping.fields ?? []
            var confirmed: [String: ExtractChoice] = [:]
            for field in fields where confirmed.count < 2 {
                guard let first = field.candidates.first, let last = field.candidates.last else { continue }
                switch field.field.kind {
                case .number where !confirmed.values.contains { if case .value = $0 { true } else { false } }:
                    // As a person reading the example's correction would: the later of two values.
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
            let pass =
                confirmed.count == 2 && store.state.confirmations == confirmed && exported == confirmed
                && document.input.text == example && document.model?.fields.count == document.form.fields.count
                && document.rules != nil
            return (
                pass,
                "confirmed=\(confirmed) export=\(path) modelFields=\(document.model?.fields.count ?? 0) "
                    + "cues=\(document.rules?.correctionCues.map(\.word) ?? [])"
            )
        }
    }

    private func finish(metricsBefore: Int) async {
        for id in sessions {
            let source = AgentViewModel.transcriptsDirectory().appending(path: "\(id).md")
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
