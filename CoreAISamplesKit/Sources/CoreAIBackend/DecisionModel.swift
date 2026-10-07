import Foundation
import Tokenizers

public actor DecisionModel {
    public let bundle: DecisionBundle
    private let tokenizer: any Tokenizer
    private let lexical: DecisionLexical
    private var runtime: DecisionRuntime?
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    private init(
        bundle: DecisionBundle, tokenizer: any Tokenizer, lexical: DecisionLexical, runtime: DecisionRuntime
    ) {
        self.bundle = bundle
        self.tokenizer = tokenizer
        self.lexical = lexical
        self.runtime = runtime
    }

    public static func load(bundle url: URL) async throws -> DecisionModel {
        let bundle = try DecisionBundle(url: url)
        let lexical = try DecisionLexical(url: bundle.lexicalURL, width: bundle.manifest.backbone.hiddenSize)
        let tokenizer = try await AutoTokenizer.from(modelFolder: bundle.tokenizerURL)
        try Task.checkCancellation()
        let runtime = try await DecisionRuntime.load(bundle: bundle, capacity: 256)
        try Task.checkCancellation()
        return DecisionModel(bundle: bundle, tokenizer: tokenizer, lexical: lexical, runtime: runtime)
    }

    public func tokenCount(for request: DecisionRequest) throws -> Int {
        try DecisionRenderer.render(request, tokenizer: tokenizer, limits: bundle.limits).tokenIDs.count
    }

    public func decide(_ request: DecisionRequest) async throws -> DecisionResult {
        let started = ContinuousClock.now
        let rendered = try DecisionRenderer.render(request, tokenizer: tokenizer, limits: bundle.limits)
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        runtime = nil
        let capacity = (rendered.tokenIDs.count + 255) / 256 * 256
        var current = try await DecisionRuntime.load(bundle: bundle, capacity: capacity)
        let hidden = try await current.forward(rendered.tokenIDs[...])
        let vectors = try lexical.vectors(rendered: rendered, questions: current.questions, options: current.options)
        let logits = try await current.predict(hidden: hidden, rendered: rendered, lexical: vectors)
        let answers = try DecisionResult.answers(
            request: request, rendered: rendered, logits: logits, stride: current.options)
        try Task.checkCancellation()
        runtime = current
        return DecisionResult(
            answers: answers, model: bundle.name, modelVersion: bundle.version, inputTokens: rendered.tokenIDs.count,
            elapsedSeconds: (ContinuousClock.now - started) / .seconds(1))
    }

    private func acquire() async {
        if busy { await withCheckedContinuation { waiters.append($0) } }
        busy = true
    }

    private func release() {
        if waiters.isEmpty {
            busy = false
        } else {
            waiters.removeFirst().resume()
        }
    }
}
