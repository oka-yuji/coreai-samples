import CoreAIBackend
import DecideFeature
import Foundation

public enum DecideServiceError: Error, LocalizedError, Sendable {
    case noModel
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .noModel: "Open a decision bundle first."
        case .invalidInput(let message): message
        }
    }
}

@MainActor
public final class LiveDecideService: DecideService {
    private var model: DecisionModel?
    private var context: DecisionContext?
    private var documentBytes: Data?
    private let files = DecideFiles()
    private let examplesURL: URL?
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init(examplesURL: URL? = nil) {
        self.examplesURL = examplesURL
    }

    public func examples() async throws -> [DecideExample] {
        guard let examplesURL else {
            throw DecideServiceError.invalidInput("Bundled example documents were not found.")
        }
        let examples = try await files.examples(at: examplesURL)
        try Task.checkCancellation()
        return examples
    }

    public func load(path: String) async throws -> DecideBundleInfo {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        model = nil
        context = nil
        documentBytes = nil
        let loaded = try await DecisionModel.load(bundle: URL(filePath: path))
        try Task.checkCancellation()
        model = loaded
        let bundle = await loaded.bundle
        return DecideBundleInfo(name: bundle.name, path: bundle.url.path, tokenLimit: bundle.limits.tokens)
    }

    public func count(_ input: DecideInput) async throws -> Int {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        guard let model else { throw DecideServiceError.noModel }
        let count = try await model.tokenCount(for: Self.request(input))
        try Task.checkCancellation()
        return count
    }

    public func read(_ url: URL) async throws -> String {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        let text = try await files.read(url)
        try Task.checkCancellation()
        return text
    }

    public func waitForIdle() async {
        await acquire()
        release()
    }

    public func decide(_ input: DecideInput, reuse: Bool) async throws -> DecideOutput {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        guard let model else { throw DecideServiceError.noModel }
        let request = try Self.request(input)
        let result: DecisionResult
        var reused = false
        var contextSeconds = 0.0
        if reuse {
            let bytes = Data(input.document.utf8)
            if context == nil || documentBytes != bytes {
                context = nil
                documentBytes = nil
                let created = try await model.context(state: input.document)
                try Task.checkCancellation()
                context = created
                documentBytes = bytes
                contextSeconds = created.creationSeconds
            } else {
                reused = true
            }
            guard let context else { throw DecideServiceError.noModel }
            result = try await context.decide(request.questions)
        } else {
            context = nil
            documentBytes = nil
            result = try await model.decide(request)
        }
        try Task.checkCancellation()
        return try Self.output(result, input: input, request: request, reused: reused, contextSeconds: contextSeconds)
    }

    public static func request(_ input: DecideInput) throws -> DecisionRequest {
        if let problem = input.problem { throw DecideServiceError.invalidInput(problem) }
        return DecisionRequest(
            state: input.document,
            questions: input.items.map { item in
                let criteria: DecisionQuestion.Criteria
                switch item.kind {
                case .choice:
                    criteria = .choice(
                        item.options.map {
                            DecisionOption(id: $0.key, description: $0.text.isEmpty ? nil : $0.text)
                        })
                case .noul:
                    criteria = .noul([
                        "true": item.options[0].text.isEmpty ? nil : item.options[0].text,
                        "false": item.options[1].text.isEmpty ? nil : item.options[1].text,
                    ])
                case .score:
                    criteria = .score(item.options.map { $0.text.isEmpty ? nil : $0.text })
                }
                return DecisionQuestion(
                    id: item.key, instructions: item.instructions.isEmpty ? nil : item.instructions, criteria: criteria)
            })
    }

    private static func output(
        _ result: DecisionResult, input: DecideInput, request: DecisionRequest, reused: Bool, contextSeconds: Double
    ) throws -> DecideOutput {
        guard result.answers.count == input.items.count else {
            throw DecideServiceError.invalidInput("The model returned a different number of items.")
        }
        let rows = zip(result.answers, input.items).map { answer, item in
            let probabilities = answer.distribution.enumerated().map { index, probability in
                let description: String
                switch item.kind {
                case .choice:
                    description = item.options.first { $0.key.utf8.elementsEqual(probability.option.utf8) }?.text ?? ""
                case .noul:
                    description = item.options[probability.option == "true" ? 0 : 1].text
                case .score:
                    description =
                        Int(probability.option).flatMap {
                            item.options.indices.contains($0) ? item.options[$0].text : nil
                        } ?? ""
                }
                return DecideProbability(
                    id: index, option: probability.option,
                    label: description.isEmpty ? probability.option : probability.option + " — " + description,
                    probability: probability.probability)
            }
            let expected: Float?
            if case .score(_, _, let value) = answer { expected = value } else { expected = nil }
            return DecideRow(
                id: item.id, key: item.key, candidate: answer.top, probabilities: probabilities,
                reviewFirst: result.reviewFirstQuestions.contains { $0.utf8.elementsEqual(answer.id.utf8) },
                expectedScore: expected)
        }
        return DecideOutput(
            rows: rows, modelName: result.model, modelVersion: result.modelVersion,
            inputTokens: result.inputTokens, reusedTokens: result.reusedTokens, seconds: result.elapsedSeconds,
            contextSeconds: contextSeconds, reusedDocument: reused, recomputedAll: result.recomputedAll,
            requestJSON: String(decoding: request.jsonData(), as: UTF8.self),
            responseJSON: String(decoding: try result.systemOneJSON(for: request), as: UTF8.self))
    }

    private func acquire() async {
        if busy { await withCheckedContinuation { waiters.append($0) } }
        busy = true
    }

    private func release() {
        if waiters.isEmpty { busy = false } else { waiters.removeFirst().resume() }
    }
}

public actor DecideFiles {
    public init() {}

    public func examples(at url: URL) throws -> [DecideExample] {
        try JSONDecoder().decode([DecideExample].self, from: Data(contentsOf: url))
    }

    public func read(_ url: URL) throws -> String {
        guard url.isFileURL, ["txt", "md", "markdown"].contains(url.pathExtension.lowercased()) else {
            throw DecideServiceError.invalidInput("Choose a UTF-8 .txt or .md document.")
        }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let resolved = url.resolvingSymlinksInPath()
        guard try resolved.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else {
            throw DecideServiceError.invalidInput("Choose a regular text document.")
        }
        let handle = try FileHandle(forReadingFrom: resolved)
        defer { try? handle.close() }
        let limit = 4 * 1_024 * 1_024
        var data = Data()
        while data.count <= limit {
            try Task.checkCancellation()
            let chunk = try handle.read(upToCount: min(65_536, limit + 1 - data.count)) ?? Data()
            if chunk.isEmpty { break }
            data.append(chunk)
        }
        guard data.count <= limit else {
            throw DecideServiceError.invalidInput("Choose a document smaller than 4 MiB.")
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw DecideServiceError.invalidInput("The document must use UTF-8 text encoding.")
        }
        return text
    }

    public func write(_ data: Data, to url: URL) throws {
        guard url.isFileURL else { throw DecideServiceError.invalidInput("Choose a local export file.") }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }
}
