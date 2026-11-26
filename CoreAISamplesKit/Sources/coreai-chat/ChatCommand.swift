import CoreAIBackend
import Foundation
import FoundationModels

@main
enum ChatCommand {
    static let usage =
        """
        usage: coreai-chat --model <bundle> --prompt <text> [--prompt <text> ...] [--system <text>] [--max-tokens <n>] [--stats] [--recover-prepared-copy]
               coreai-chat --model <bundle> --clear-prepared-copy

        """

    struct LoadReport: Encodable {
        let cached: Bool
        let cacheQueryError: String?
        let recoveredPreparedCopy: Bool?
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
        var recoverPreparedCopy = false
        var clearPreparedCopy = false
        var arguments = CommandLine.arguments.dropFirst().makeIterator()
        while let argument = arguments.next() {
            switch argument {
            case "--model": model = arguments.next()
            case "--prompt": prompts += arguments.next().map { [$0] } ?? []
            case "--system": system = arguments.next()
            case "--max-tokens": maxTokens = arguments.next().flatMap(Int.init) ?? 0
            case "--stats": stats = true
            case "--recover-prepared-copy": recoverPreparedCopy = true
            case "--clear-prepared-copy": clearPreparedCopy = true
            default: maxTokens = 0
            }
        }
        let hasInput = clearPreparedCopy ? CommandLine.arguments.count == 4 : !prompts.isEmpty
        guard let model, hasInput, maxTokens > 0 else {
            FileHandle.standardError.write(Data(usage.utf8))
            exit(2)
        }
        do {
            let url = URL(filePath: model)
            if clearPreparedCopy {
                try Engine.deleteSpecializationCache(bundle: ModelBundle(url: url))
                return
            }
            try await run(
                model: url, prompts: prompts, system: system, maxTokens: maxTokens, stats: stats,
                recoverPreparedCopy: recoverPreparedCopy)
        } catch {
            let message: String
            if let failure = error as? PreparedCopyError {
                let hint = failure == .recoveryFailed ? " Use --clear-prepared-copy with --model." : ""
                message = "\(failure): \(failure.localizedDescription)\(hint)"
            } else {
                message = String(describing: error)
            }
            FileHandle.standardError.write(Data("error: \(message)\n".utf8))
            exit(1)
        }
    }

    static func run(
        model url: URL, prompts: [String], system: String?, maxTokens: Int, stats: Bool, recoverPreparedCopy: Bool
    ) async throws {
        let clock = ContinuousClock()
        let start = clock.now
        let model = try await LocalLanguageModel(bundle: url, recoverPreparedCopy: recoverPreparedCopy)
        if stats {
            report(
                LoadReport(
                    cached: model.loadStats.cached, cacheQueryError: model.loadStats.cacheQueryError,
                    recoveredPreparedCopy: recoverPreparedCopy ? model.loadStats.recoveredPreparedCopy : nil,
                    specializeS: model.loadStats.specializeSeconds, loadS: (clock.now - start) / .seconds(1)))
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
            for try await snapshot in session.streamResponse(
                to: prompt, options: GenerationOptions(maximumResponseTokens: maxTokens))
            {
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
            report(
                TurnReport(
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
