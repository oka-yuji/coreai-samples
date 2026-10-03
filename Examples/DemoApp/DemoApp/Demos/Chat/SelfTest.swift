import Foundation

enum SelfTest {
    static func run() -> Never {
        Task { @MainActor in exit(await execute()) }
        dispatchMain()
    }

    @MainActor
    static func execute() async -> Int32 {
        guard let model = LaunchArguments.value("--model"), let file = LaunchArguments.value("--prompts-file"),
            let text = try? String(contentsOfFile: file, encoding: .utf8)
        else {
            log("usage: DemoApp --selftest --model <folder> --prompts-file <file> [--max-tokens N] [--context-limit N]")
            return finish(false)
        }
        let prompts = text.components(separatedBy: "\n===PROMPT===\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let chat = ChatViewModel()
        if let maxTokens = LaunchArguments.value("--max-tokens").flatMap(Int.init) { chat.maxTokens = maxTokens }
        await chat.load(URL(filePath: model, directoryHint: .isDirectory))
        guard chat.phase == .ready, let stats = chat.loadStats else {
            log("[selftest] load failed: \(chat.phase)")
            return finish(false)
        }
        log(String(
            format: "[selftest] model=%@ prompts=%d maxTokens=%d contextLimit=%@ cached=%@ specializeSeconds=%.2f loadSeconds=%.2f",
            chat.modelName, prompts.count, chat.maxTokens, chat.contextLimit.map(String.init) ?? "default",
            String(stats.cached), stats.specializeSeconds, chat.loadSeconds))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var pass = !prompts.isEmpty
        for prompt in prompts {
            chat.input = prompt
            guard chat.send() else {
                log("[selftest] send refused: phase=\(chat.phase) contextFull=\(chat.contextFull)")
                pass = false
                break
            }
            await chat.generation?.value
            guard let result = chat.last else {
                pass = false
                break
            }
            FileHandle.standardOutput.write((try? encoder.encode(result)) ?? Data())
            FileHandle.standardOutput.write(Data("\n".utf8))
            pass = pass && result.error == nil && ["stopToken", "maxTokens"].contains(result.stopReason) && result.outputTokens > 0
                && (result.turn == 1 || (result.cachedTokens ?? 0) > 0)
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
