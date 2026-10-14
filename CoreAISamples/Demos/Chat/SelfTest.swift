import ConversationFeature
import Foundation
import RAGContext

enum SelfTest {
    static func run() -> Never {
        Task { @MainActor in exit(await execute()) }
        RunLoop.main.run()
        fatalError("the main run loop returned")
    }

    static func isOff(_ value: String?) -> Bool {
        ["0", "no", "off", "false"].contains(value ?? "")
    }

    @MainActor
    static func execute() async -> Int32 {
        let questions = LaunchArguments.value("--rag").map { $0.split(separator: "|").map(String.init) } ?? []
        let file = LaunchArguments.value("--prompts-file").flatMap { try? String(contentsOfFile: $0, encoding: .utf8) }
        guard let model = LaunchArguments.value("--model"), !questions.isEmpty || file != nil,
            LaunchArguments.value("--rag-budget") == nil || ChatViewModel.launchRAGBudget != nil,
            questions.isEmpty || !isOff(LaunchArguments.value("--rag-web"))
                || !isOff(LaunchArguments.value("--rag-local")),
            LaunchArguments.value("--effort").map({ AgentReasoningEffort(argument: $0) != nil }) ?? true
        else {
            log(
                "usage: CoreAISamples --selftest --model <folder> (--prompts-file <file> | --rag \"question 1|question 2\" [--rag-web off] [--rag-local off] [--rag-docs <folder>] [--rag-budget 3000|6000]) [--max-tokens N] [--context-limit N] [--effort off|low|medium|xhigh]"
            )
            return finish(false)
        }
        let prompts =
            questions.isEmpty
            ? (file ?? "").components(separatedBy: "\n===PROMPT===\n").map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
            }.filter { !$0.isEmpty }
            : questions
        let chat = ChatViewModel()
        let preferences = ConversationPreferences(persists: false)
        preferences.resetForHeadless()
        preferences.effort = LaunchArguments.value("--effort").flatMap(AgentReasoningEffort.init(argument:)) ?? .medium
        if let maxTokens = LaunchArguments.value("--max-tokens").flatMap(Int.init) { preferences.maxTokens = maxTokens }
        if !questions.isEmpty {
            chat.ragEnabled = true
            chat.ragWeb = !isOff(LaunchArguments.value("--rag-web"))
            chat.ragLocal = !isOff(LaunchArguments.value("--rag-local"))
        }
        await chat.load(URL(filePath: model, directoryHint: .isDirectory))
        guard chat.phase == .ready, let stats = chat.loadStats else {
            log("[selftest] load failed: \(chat.phase)")
            return finish(false)
        }
        let host = ConversationHost(chat: chat, preferences: preferences)
        log(
            String(
                format:
                    "[selftest] model=%@ prompts=%d maxTokens=%d contextLimit=%d effort=%@ cached=%@ specializeSeconds=%.2f loadSeconds=%.2f",
                chat.modelName, prompts.count, TurnPolicy(.chat, settings: preferences.settings).maxTokens,
                chat.contextLimit, preferences.effort.label, String(stats.cached), stats.specializeSeconds,
                chat.loadSeconds))
        if chat.ragEnabled {
            let settings = RAGSettings.fromDefaults()
            log(
                "[rag] mode=on web=\(chat.ragWeb) local=\(chat.ragLocal) search=\(settings.provider) results=\(settings.searchResults) "
                    + "pages=\(settings.pagesToRead) pageBudget=\(settings.pageCharacterBudget)chars sourceBudget=\(chat.ragBudget)tok "
                    + "promptBudget=\(RAGContext.promptTokenBudget)tok today=\(RAGContext.today())")
            log("[rag] documents=" + settings.documentsDirectory.path(percentEncoded: false))
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var pass = !prompts.isEmpty
        for (index, prompt) in prompts.enumerated() {
            chat.input = prompt
            guard host.submit() else {
                log("[selftest] send refused: phase=\(chat.phase) contextFull=\(host.contextFull || chat.contextFull)")
                pass = false
                break
            }
            while host.state.isRunning || chat.isResponding { try? await Task.sleep(for: .milliseconds(20)) }
            guard let result = chat.ragEnabled ? chat.last : host.runner.session.lastGeneration else {
                log(
                    "[selftest] no result: ragIsOn=\(chat.ragIsOn) running=\(host.state.isRunning) "
                        + "failure=\(host.state.turns.last?.failure ?? "none")")
                pass = false
                break
            }
            if chat.ragEnabled, let retrieval = chat.retrieval {
                log("[rag] turn=\(index) question=\(prompt)")
                log("[rag] turn=\(index) \(retrieval.summary)")
                for line in retrieval.notes { log("[rag] turn=\(index) note \(line)") }
                for line in retrieval.citations { log("[rag] turn=\(index) source \(line)") }
                for (number, source) in retrieval.sources.enumerated() {
                    log("[rag] turn=\(index) passage [\(number + 1)] \(source.text.replacing("\n", with: "\\n"))")
                }
                let thinking = chat.ragTurns.last?.reasoning ?? ""
                log("[rag] turn=\(index) thinking (\(thinking.count) chars)=\(thinking.replacing("\n", with: "\\n"))")
                log("[rag] turn=\(index) answer=\(result.answer.replacing("\n", with: " "))")
            }
            FileHandle.standardOutput.write((try? encoder.encode(result)) ?? Data())
            FileHandle.standardOutput.write(Data("\n".utf8))
            log("[selftest] \(result.statusLine)")
            log("[selftest] engineTokens=\(await chat.model?.processedTokens().count ?? 0)")
            let generated =
                result.error == nil && ["stopToken", "maxTokens"].contains(result.stopReason) && result.outputTokens > 0
            if !chat.ragEnabled {
                log("[selftest] sessionTurn=\(result.turn) expected=\(index + 1)")
                let continued = result.turn == index + 1 && (result.turn == 1 || (result.cachedTokens ?? 0) > 0)
                pass = pass && generated && continued
            } else if (result.sources ?? 0) > 0 {
                pass = pass && generated && !result.answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            } else {
                pass =
                    pass && result.stopReason == "noSources" && result.answer == "Could not retrieve any sources."
                    && result.inputTokens == nil
            }
        }
        await MetricsLog.flush()
        return finish(pass)
    }

    private static func finish(_ pass: Bool) -> Int32 {
        FileHandle.standardOutput.write(Data((pass ? "SELFTEST PASS\n" : "SELFTEST FAIL\n").utf8))
        return pass ? 0 : 1
    }

    private static func log(_ line: String) {
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }
}
