import CryptoKit
import Foundation
import Tokenizers

public actor DecisionModel {
    public let bundle: DecisionBundle
    private let tokenizer: any Tokenizer
    private let lexical: DecisionLexical
    private let reviewPolicy: DecisionReviewPolicy
    private var runtime: DecisionRuntime?
    private var document: CachedDecisionDocument?
    private var parkedStates: DecisionRuntime.RetainedStates?
    private var lastQueryRows: Int?
    private var lastHeadRows: Int?
    private var needsRuntimeReload = false
    private var contextGeneration = 0
    private var decisionCheckpoint: (@Sendable () async throws -> Void)?
    private var reloadCheckpoint: (@Sendable () async throws -> Void)?
    private var queuedCheckpoint: (@Sendable () -> Void)?
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    private init(
        bundle: DecisionBundle, tokenizer: any Tokenizer, lexical: DecisionLexical, runtime: DecisionRuntime,
        reviewPolicy: DecisionReviewPolicy
    ) {
        self.bundle = bundle
        self.tokenizer = tokenizer
        self.lexical = lexical
        self.runtime = runtime
        self.reviewPolicy = reviewPolicy
    }

    public static func load(bundle url: URL) async throws -> DecisionModel {
        let bundle = try DecisionBundle(url: url)
        let lexical = try DecisionLexical(url: bundle.lexicalURL, width: bundle.manifest.backbone.hiddenSize)
        let tokenizer = try await AutoTokenizer.from(modelFolder: bundle.tokenizerURL)
        try Task.checkCancellation()
        let runtime = try await DecisionRuntime.load(bundle: bundle, capacity: 256)
        try Task.checkCancellation()
        return try DecisionModel(
            bundle: bundle, tokenizer: tokenizer, lexical: lexical, runtime: runtime,
            reviewPolicy: DecisionReviewPolicy.bundled())
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
        document = nil
        parkedStates = nil
        lastQueryRows = nil
        lastHeadRows = nil
        needsRuntimeReload = false
        contextGeneration += 1
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
            elapsedSeconds: (ContinuousClock.now - started) / .seconds(1),
            reviewFirstQuestions: answers.filter { reviewPolicy.requestsReview($0) }.map(\.id))
    }

    private func acquire() async {
        if busy {
            await withCheckedContinuation {
                waiters.append($0)
                queuedCheckpoint?()
            }
        }
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

extension DecisionModel {
    public func context(
        state: String, questionBudget: Int = 1_024, overflow: DecisionOverflowPolicy = .reject
    ) async throws -> DecisionContext {
        let started = ContinuousClock.now
        guard questionBudget > 0, questionBudget < bundle.limits.tokens else {
            throw DecisionError.invalidRequest("The question budget must be positive and below the context limit.")
        }
        let request = DecisionRequest(state: state, questions: [])
        let prefix = DecisionRenderer.fragments(for: request).prefix(2).flatMap {
            tokenizer.encode(text: $0.text, addSpecialTokens: false)
        }
        guard prefix.count + questionBudget <= bundle.limits.tokens else {
            throw DecisionError.tooManyTokens(count: prefix.count + questionBudget, limit: bundle.limits.tokens)
        }
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        document = nil
        parkedStates = nil
        lastQueryRows = nil
        lastHeadRows = nil
        needsRuntimeReload = false
        runtime = nil
        contextGeneration += 1
        let generation = contextGeneration
        var current = try await DecisionRuntime.load(
            bundle: bundle, capacity: (prefix.count + questionBudget + 255) / 256 * 256)
        let hidden = try await current.forward(prefix[...])
        let checkpoint = current.checkpoint()
        try Task.checkCancellation()
        runtime = current
        document = CachedDecisionDocument(
            generation: generation, state: state, prefix: prefix, hidden: hidden,
            checkpoint: checkpoint, questionBudget: questionBudget, previous: nil)
        let tokenBytes = prefix.map { Int64($0).littleEndian }.withUnsafeBufferPointer { Data(buffer: $0) }
        let key =
            bundle.version + ":" + String(bundle.manifest.rendering.version) + ":"
            + SHA256.hash(data: tokenBytes).map { String(format: "%02x", $0) }.joined()
        return DecisionContext(
            model: self, generation: generation, prefixTokenCount: prefix.count, questionBudget: questionBudget,
            key: key, creationSeconds: (ContinuousClock.now - started) / .seconds(1))
    }

    func decide(questions: [DecisionQuestion], generation: Int) async throws -> DecisionResult {
        let started = ContinuousClock.now
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        guard let saved = document, saved.generation == generation else { throw DecisionError.contextInvalidated }
        let request = DecisionRequest(state: saved.state, questions: questions)
        let rendered = try DecisionRenderer.render(request, limits: bundle.limits, prefix: saved.prefix) {
            tokenizer.encode(text: $0, addSpecialTokens: false)
        }
        guard rendered.tokenIDs.starts(with: saved.prefix) else { throw DecisionError.contextInvalidated }
        let questionTokens = rendered.tokenIDs.count - saved.prefix.count
        guard questionTokens <= saved.questionBudget else {
            throw DecisionError.questionBudgetExceeded(count: questionTokens, limit: saved.questionBudget)
        }
        let headRows = (rendered.tokenIDs.count + 511) / 512 * 512
        try await prepareReusableRuntime(queryRows: questionTokens, headRows: headRows)
        try Task.checkCancellation()
        guard var current = runtime else { throw DecisionError.contextInvalidated }
        runtime = nil
        var completed = false
        defer {
            runtime = current
            if !completed { needsRuntimeReload = true }
        }
        try current.restore(saved.checkpoint, processed: saved.prefix.count)
        let rest = try await current.forward(rendered.tokenIDs.dropFirst(saved.prefix.count))
        lastQueryRows = questionTokens
        if let checkpoint = decisionCheckpoint { try await checkpoint() }
        try Task.checkCancellation()
        let vectors = try lexical.vectors(rendered: rendered, questions: current.questions, options: current.options)
        let logits = try await current.predict(hidden: saved.hidden, rest: rest, rendered: rendered, lexical: vectors)
        lastHeadRows = headRows
        let answers = try DecisionResult.answers(
            request: request, rendered: rendered, logits: logits, stride: current.options)
        try Task.checkCancellation()
        let result = DecisionResult(
            answers: answers, model: bundle.name, modelVersion: bundle.version, inputTokens: rendered.tokenIDs.count,
            elapsedSeconds: (ContinuousClock.now - started) / .seconds(1),
            reusedTokens: saved.prefix.count, recomputedAll: true,
            changedQuestions: DecisionResult.changedQuestions(
                previous: saved.previous, answers: answers, policy: reviewPolicy),
            reviewFirstQuestions: answers.filter { reviewPolicy.requestsReview($0) }.map(\.id))
        document?.previous = result
        completed = true
        return result
    }

    private func prepareReusableRuntime(queryRows: Int, headRows: Int) async throws {
        func changesBand(_ previous: Int?, _ next: Int) -> Bool {
            guard let previous else { return false }
            return max(previous, next) >= min(previous, next) * 2
        }
        if runtime != nil
            && (needsRuntimeReload || changesBand(lastQueryRows, queryRows) || changesBand(lastHeadRows, headRows))
        {
            parkRuntime()
        }
        if runtime == nil {
            guard let parkedStates else { throw DecisionError.contextInvalidated }
            if let checkpoint = reloadCheckpoint { try await checkpoint() }
            runtime = try await DecisionRuntime.load(
                bundle: bundle, capacity: parkedStates.capacity, retaining: parkedStates)
            self.parkedStates = nil
            lastQueryRows = nil
            lastHeadRows = nil
            needsRuntimeReload = false
        }
    }

    private func parkRuntime() {
        if let current = runtime { parkedStates = current.retainedStates() }
        runtime = nil
    }

    func setReloadCheckpoint(_ checkpoint: (@Sendable () async throws -> Void)?) {
        reloadCheckpoint = checkpoint
    }

    func setQueuedCheckpoint(_ checkpoint: (@Sendable () -> Void)?) {
        queuedCheckpoint = checkpoint
    }

    func setDecisionCheckpoint(_ checkpoint: (@Sendable () async throws -> Void)?) {
        decisionCheckpoint = checkpoint
    }
}
