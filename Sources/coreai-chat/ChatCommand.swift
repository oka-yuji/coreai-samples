import CoreAIBackend
import Foundation

@main
enum ChatCommand {
    static let usage = "usage: coreai-chat --model <bundle> --prompt <text> [--max-tokens <n>] [--stats]\n"

    struct Report: Encodable {
        let cached: Bool
        let specializeS: Double
        let loadS: Double
        let promptTokens: Int
        let prefillS: Double
        let prefillTokS: Double
        let outputTokens: Int
        let stop: String
        let decodeS: Double
        let decodeTokS: Double?
        let totalS: Double
    }

    static func main() async {
        var model: String?
        var prompt: String?
        var maxTokens = 512
        var stats = false
        var arguments = CommandLine.arguments.dropFirst().makeIterator()
        while let argument = arguments.next() {
            switch argument {
            case "--model": model = arguments.next()
            case "--prompt": prompt = arguments.next()
            case "--max-tokens": maxTokens = arguments.next().flatMap(Int.init) ?? 0
            case "--stats": stats = true
            default: maxTokens = 0
            }
        }
        guard let model, let prompt, maxTokens > 0 else {
            FileHandle.standardError.write(Data(usage.utf8))
            exit(2)
        }
        do {
            try await run(model: model, prompt: prompt, maxTokens: maxTokens, stats: stats)
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }

    static func run(model: String, prompt: String, maxTokens: Int, stats: Bool) async throws {
        let bundle = try ModelBundle(url: URL(filePath: model))
        let tokenizer = try await bundle.loadTokenizer()
        let engine = try await Engine(bundle: bundle)
        let input = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": prompt]]).map(Int32.init)
        let stopTokens = Set([tokenizer.eosTokenId].compactMap { $0.map(Int32.init) })

        var pending: [Int] = []
        let start = ContinuousClock.now
        let output = try await engine.generate(input, maxTokens: maxTokens, stopTokens: stopTokens) { token in
            guard !stopTokens.contains(token) else { return }
            pending.append(Int(token))
            let text = tokenizer.decode(tokens: pending)
            if !text.hasSuffix("\u{FFFD}") {
                FileHandle.standardOutput.write(Data(text.utf8))
                pending.removeAll()
            }
        }
        let total = (ContinuousClock.now - start) / .seconds(1)
        FileHandle.standardOutput.write(Data((tokenizer.decode(tokens: pending) + "\n").utf8))

        guard stats else { return }
        let prefill = engine.stats.prefillSeconds
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = .sortedKeys
        let report = Report(
            cached: engine.loadStats.cached,
            specializeS: engine.loadStats.specializeSeconds,
            loadS: engine.loadStats.totalSeconds,
            promptTokens: input.count,
            prefillS: prefill,
            prefillTokS: Double(input.count) / prefill,
            outputTokens: output.count,
            stop: stopTokens.contains(output.last ?? -1) ? "eos" : "max",
            decodeS: total - prefill,
            decodeTokS: output.count > 1 ? Double(output.count - 1) / (total - prefill) : nil,
            totalS: total)
        FileHandle.standardError.write(try encoder.encode(report) + Data("\n".utf8))
    }
}
