import CoreAIBackend
import Foundation
import FoundationModels

@main
enum ChatCommand {
    static let usage =
        "usage: coreai-chat --model <bundle> --prompt <text> [--prompt <text> ...] [--system <text>] [--max-tokens <n>] [--stats]\n"

    struct LoadReport: Encodable {
        let cached: Bool
        let specializeS: Double
        let loadS: Double
    }

    struct TurnReport: Encodable {
        let turn: Int
        let inputTokens: Int
        let cachedTokens: Int
        let outputTokens: Int
        let reasoningTokens: Int
        let stopReason: String?
        let totalS: Double
    }

    static func main() async {
        var model: String?
        var prompts: [String] = []
        var system: String?
        var maxTokens = 512
        var stats = false
        var arguments = CommandLine.arguments.dropFirst().makeIterator()
        while let argument = arguments.next() {
            switch argument {
            case "--model": model = arguments.next()
            case "--prompt": prompts += arguments.next().map { [$0] } ?? []
            case "--system": system = arguments.next()
            case "--max-tokens": maxTokens = arguments.next().flatMap(Int.init) ?? 0
            case "--stats": stats = true
            default: maxTokens = 0
            }
        }
        guard let model, !prompts.isEmpty, maxTokens > 0 else {
            FileHandle.standardError.write(Data(usage.utf8))
            exit(2)
        }
        do {
            try await run(model: model, prompts: prompts, system: system, maxTokens: maxTokens, stats: stats)
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }

    static func run(model path: String, prompts: [String], system: String?, maxTokens: Int, stats: Bool) async throws {
        let clock = ContinuousClock()
        let start = clock.now
        let model = try await LocalLanguageModel(bundle: URL(filePath: path))
        if stats {
            report(LoadReport(cached: model.loadStats.cached, specializeS: model.loadStats.specializeSeconds, loadS: (clock.now - start) / .seconds(1)))
        }
        let session = LanguageModelSession(model: model, instructions: system)
        for (index, prompt) in prompts.enumerated() {
            write("\(index == 0 ? "" : "\n")> \(prompt)\n")
            let base = session.transcript.count
            let before = session.usage
            let turnStart = clock.now
            var printed = 0
            var shown = false
            func showReasoning() {
                guard !shown else { return }
                shown = true
                let reasoning = session.transcript.dropFirst(base).compactMap { entry -> String? in
                    guard case .reasoning(let reasoning) = entry else { return nil }
                    return text(reasoning.segments)
                }.joined()
                if !reasoning.isEmpty { write("[reasoning]\n\(reasoning)\n[response]\n") }
            }
            for try await snapshot in session.streamResponse(to: prompt, options: GenerationOptions(maximumResponseTokens: maxTokens)) {
                showReasoning()
                let content = snapshot.content.utf8
                write(String(decoding: content.dropFirst(printed), as: UTF8.self))
                printed = content.count
            }
            showReasoning()
            write("\n")
            guard stats else { continue }
            var stopReason: String?
            if case .response(let response) = session.transcript.last {
                stopReason = try? response.metadata["stopReason"]?.value(String.self)
            }
            let usage = session.usage
            report(TurnReport(
                turn: index + 1,
                inputTokens: usage.input.totalTokenCount - before.input.totalTokenCount,
                cachedTokens: usage.input.cachedTokenCount - before.input.cachedTokenCount,
                outputTokens: usage.output.totalTokenCount - before.output.totalTokenCount,
                reasoningTokens: usage.output.reasoningTokenCount - before.output.reasoningTokenCount,
                stopReason: stopReason,
                totalS: (clock.now - turnStart) / .seconds(1)))
        }
    }

    static func text(_ segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment -> String? in
            if case .text(let text) = segment { text.content } else { nil }
        }.joined()
    }

    static func write(_ text: String) {
        FileHandle.standardOutput.write(Data(text.utf8))
    }

    static func report(_ value: some Encodable) {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = .sortedKeys
        FileHandle.standardError.write(((try? encoder.encode(value)) ?? Data()) + Data("\n".utf8))
    }
}
