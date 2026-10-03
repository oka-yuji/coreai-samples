import AppKit
import Foundation
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

    private var attached = false
    private var results: [StepResult] = []
    private var lines: [String] = []
    private var outDir = URL.temporaryDirectory
    private var maxStall = 0.0

    func attach(navigator: DemoNavigator, chat: ChatViewModel, models: ModelsViewModel) {
        guard let path = LaunchArguments.value("--gui-e2e"), !attached else { return }
        attached = true
        outDir = URL(filePath: path, directoryHint: .isDirectory)
        Task { await run(navigator: navigator, chat: chat, models: models) }
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
                seen.insert(UInt32(color.redComponent * 255) << 16 | UInt32(color.greenComponent * 255) << 8 | UInt32(color.blueComponent * 255))
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
        guard let view = window.contentView, let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return nil }
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
        results.append(StepResult(
            step: index, title: title, pass: pass, detail: detail + " | shot: " + shot.detail,
            screenshot: shot.name, seconds: (seconds * 100).rounded() / 100))
        note(String(format: "step %d %@ %@ %.1fs %@", index, slug, pass ? "PASS" : "FAIL", seconds, Self.head(detail, 600)))
    }

    private func ask(_ chat: ChatViewModel, _ text: String, timeout: Double = 900) async -> Bool {
        chat.input = text
        guard chat.send() else {
            note("send refused: phase=\(chat.phase) responding=\(chat.isResponding) contextFull=\(chat.contextFull)")
            return false
        }
        return await wait(seconds: timeout) { !chat.isResponding }
    }

    private func run(navigator: DemoNavigator, chat: ChatViewModel, models: ModelsViewModel) async {
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let metricsBefore = (try? String(contentsOf: MetricsLog.fileURL, encoding: .utf8))?.split(separator: "\n").count ?? 0
        note("out=\(outDir.path(percentEncoded: false)) contextLimit=\(chat.contextLimit.map(String.init) ?? "default")")
        guard let path = LaunchArguments.value("--model") else {
            note("--model <folder> is required")
            await finish(metricsBefore: metricsBefore)
            return
        }
        let url = URL(filePath: path, directoryHint: .isDirectory)
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
            return (folder?.name != nil,
                "folder=\(folder?.id ?? "none") name=\(folder?.name ?? "none") clearPrepared=\(cleared) message=\(models.message ?? "")")
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
            return (loaded && chat.phase == .ready && responsive && honest, String(
                format: "phase=%@ cached=%@ specializeSeconds=%.2f loadSeconds=%.2f maxMainStall=%.3f preparingShown=%@ note=%@",
                "\(chat.phase)", stats.map { String($0.cached) } ?? "nil", stats?.specializeSeconds ?? -1, chat.loadSeconds,
                self.maxStall, String(preparingShown), chat.loadNote))
        }

        await step(4, "chat", "One turn streams to the end and the status line shows the numbers") {
            let done = await self.ask(chat, "Explain in two sentences what a context window is in a language model.")
            let last = chat.last
            let pass = done && last?.error == nil && (last?.outputTokens ?? 0) > 0 && ["stopToken", "maxTokens"].contains(last?.stopReason ?? "")
            return (pass, "status=\(chat.statusLine) thoughtChars=\(chat.turns.last?.reasoning.count ?? 0) answer=\(Self.head(last?.answer ?? ""))")
        }

        await step(5, "generating", "A long reply is streaming") {
            chat.input = "Write a detailed essay of at least 1500 words about the history of computing, from the abacus to modern AI. Use multiple sections with headings."
            guard chat.send() else { return (false, "send refused") }
            let grew = await self.wait(seconds: 300) {
                guard let turn = chat.turns.last, chat.turns.count == 2 else { return false }
                return turn.reasoning.count + turn.response.count > 300
            }
            let turn = chat.turns.last
            return (grew && chat.isResponding,
                "responding=\(chat.isResponding) thoughtChars=\(turn?.reasoning.count ?? 0) answerChars=\(turn?.response.count ?? 0)")
        }

        await step(6, "stopped", "Stop ends the reply within three seconds and keeps what was written") {
            let started = ContinuousClock.now
            chat.stop()
            let halted = await self.wait(seconds: 3) { !chat.isResponding }
            let seconds = (ContinuousClock.now - started) / .seconds(1)
            let turn = chat.turns.last
            let kept = (turn?.reasoning.count ?? 0) + (turn?.response.count ?? 0)
            return (halted && chat.last?.stopReason == "cancelled" && kept > 0,
                String(format: "halted=%@ seconds=%.3f keptChars=%d status=%@", String(halted), seconds, kept, chat.statusLine))
        }

        await step(7, "new-chat", "New Chat starts an empty conversation") {
            chat.newChat()
            await self.settle()
            return (chat.entries.isEmpty && chat.last == nil && chat.phase == .ready,
                "entries=\(chat.entries.count) phase=\(chat.phase) status=\(chat.statusLine)")
        }

        if chat.contextLimit != nil {
            await step(8, "context-full", "A long chat with a small context limit ends with a notice, not an error") {
                var sent = 0
                while !chat.contextFull, sent < 40 {
                    sent += 1
                    guard await self.ask(chat, "Tell me one more fact about the history of computing, in about 80 words.") else { break }
                    self.note("context turn \(sent): \(chat.statusLine) notice=\(chat.contextNotice ?? "none")")
                }
                return (chat.contextFull && chat.contextNotice != nil && chat.phase == .ready && chat.last?.error == nil,
                    "turns=\(sent) contextFull=\(chat.contextFull) notice=\(chat.contextNotice ?? "none") stop=\(chat.last?.stopReason ?? "none")")
            }
        }

        await finish(metricsBefore: metricsBefore)
    }

    private func finish(metricsBefore: Int) async {
        await MetricsLog.flush()
        let metrics = ((try? String(contentsOf: MetricsLog.fileURL, encoding: .utf8)) ?? "").split(separator: "\n").dropFirst(metricsBefore)
        try? Data((metrics.joined(separator: "\n") + "\n").utf8).write(to: outDir.appending(path: "metrics.jsonl"), options: .atomic)
        note("metrics rows written by this run: \(metrics.count)")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try? encoder.encode(results).write(to: outDir.appending(path: "results.json"), options: .atomic)
        let passed = results.count { $0.pass }
        let pass = !results.isEmpty && passed == results.count
        note("GUI E2E \(pass ? "PASS" : "FAIL") \(passed)/\(results.count)")
        try? Data((lines.joined(separator: "\n") + "\n").utf8).write(to: outDir.appending(path: "driver.log"), options: .atomic)
        exit(pass ? 0 : 1)
    }
}
