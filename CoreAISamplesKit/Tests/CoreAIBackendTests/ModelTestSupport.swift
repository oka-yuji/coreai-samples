import Foundation
import FoundationModels
import Observation
import Synchronization
import Testing
import Tokenizers

@_spi(Verification) @testable import CoreAIBackend

struct ModelTestSettings {
    static let contextLimit = 16_384
    static let context = ContextOptions(reasoningLevel: .custom("off"))

    static func repetitions(_ key: String, default defaultValue: Int) throws -> Int {
        guard let value = ProcessInfo.processInfo.environment[key] else { return defaultValue }
        let count = try #require(Int(value), "\(key) must be a positive integer")
        try #require(count > 0, "\(key) must be a positive integer")
        return count
    }

    static func metadata<T: ConvertibleFromGeneratedContent>(
        _ session: LanguageModelSession, _ key: String, as type: T.Type
    ) throws -> T {
        let response = try #require(
            session.transcript.reversed().compactMap { entry -> Transcript.Response? in
                if case .response(let response) = entry { return response }
                return nil
            }.first)
        return try #require(response.metadata[key]).value(type)
    }

    static func checkState(_ model: LocalLanguageModel, label: String) async throws {
        let state = await model.processedState()
        try #require(!state.tokens.isEmpty, "\(label): the state must be populated")
        #expect(state.tokens.count <= contextLimit, "\(label): processed tokens exceed the context limit")
        let reread = try await model.resetAndRead(state.tokens)
        try #require(state.logits.count == reread.count && state.logits.count > 1)
        let check = StateCheck(current: state.logits, reread: reread)
        #expect(
            check.passed,
            "\(label): state top \(check.currentTop), reread top \(check.rereadTop), ticks \(check.marginTicks)")
        print(
            "state-check \(label) tokens=\(state.tokens.count) passed=\(check.passed) marginTicks=\(check.marginTicks)")
    }
}

struct BoundaryPrompt {
    let history: [Transcript.Entry]
    let text: String
    let tokens: [Int32]

    static func make(tokenizer: any Tokenizer, target: Int, history: [Transcript.Entry] = []) throws -> Self {
        let suffix = "\nContinue the sequence with at least ten more words."
        func render(_ repeats: Int) throws -> Self {
            let text = String(repeating: " a", count: repeats) + suffix
            let transcript = Transcript(entries: history + [.prompt(.init(segments: [.text(.init(content: text))]))])
            let messages = try Runtime.messages(transcript)
            let tokens = try tokenizer.applyChatTemplate(
                messages: messages, chatTemplate: nil, addGenerationPrompt: true, truncation: false, maxLength: nil,
                tools: nil,
                additionalContext: Runtime.templateContext(ModelTestSettings.context.reasoningLevel)
            ).map(Int32.init)
            return Self(history: history, text: text, tokens: tokens)
        }
        let empty = try render(0)
        try #require(empty.tokens.count <= target)
        var candidate = try render(target - empty.tokens.count)
        for _ in 0..<8 {
            if candidate.tokens.count == target { return candidate }
            let repeats = (candidate.text.count - suffix.count) / 2 + target - candidate.tokens.count
            try #require(repeats >= 0)
            candidate = try render(repeats)
        }
        Issue.record("Could not render exactly \(target) tokens; rendered \(candidate.tokens.count)")
        throw ModelTestError.cannotBuildBoundary
    }

    static func toolHistory() throws -> [Transcript.Entry] {
        var entries: [Transcript.Entry] = [
            .prompt(.init(segments: [.text(.init(content: "Read the three records before answering."))]))
        ]
        for index in 0..<3 {
            entries.append(
                .toolCalls(
                    .init([
                        .init(id: "call-\(index)", toolName: "read_record", arguments: try GeneratedContent(json: "{}"))
                    ])))
            entries.append(
                .toolOutput(
                    .init(
                        id: "output-\(index)", toolName: "read_record",
                        segments: [.text(.init(content: String(repeating: " record", count: 300)))])))
        }
        return entries
    }
}

enum ModelTestError: Error {
    case cannotBuildBoundary
    case noStreamFragment
    case completedBeforeCancellation
}

func responseText(in transcript: Transcript, after index: Int) -> String {
    transcript.dropFirst(index).compactMap { entry -> String? in
        guard case .response(let response) = entry else { return nil }
        return response.segments.compactMap { segment -> String? in
            if case .text(let text) = segment { return text.content }
            return nil
        }.joined()
    }.joined()
}

func cancelAfterFirstFragment(
    _ session: LanguageModelSession, prompt: String, maximumTokens: Int,
    context: ContextOptions = ModelTestSettings.context,
    beforeCancellation: @Sendable (String) async throws -> Void = { _ in },
    executionStarted: @Sendable () -> Bool = { true },
    waitUntilFinished: @Sendable () async -> Void
) async throws -> String {
    let base = session.transcript.count
    func finishExecution() async {
        if executionStarted() {
            await withTaskCancellationShield { await waitUntilFinished() }
        }
    }
    let signal = AsyncStream<String>.makeStream(bufferingPolicy: .bufferingOldest(1))
    let observing = AsyncStream<Void>.makeStream()
    let observer = Task {
        defer { observing.continuation.finish() }
        var ready = false
        for await text in Observations({ responseText(in: session.transcript, after: base) }) {
            if !ready {
                ready = true
                observing.continuation.yield(())
                observing.continuation.finish()
            }
            if !text.isEmpty {
                signal.continuation.yield(text)
                return
            }
        }
    }
    let consumer = Task {
        defer { signal.continuation.finish() }
        for await _ in observing.stream { break }
        try Task.checkCancellation()
        for try await _ in session.streamResponse(
            to: prompt, options: .init(samplingMode: .greedy, maximumResponseTokens: maximumTokens),
            contextOptions: context)
        {}
    }
    return try await withTaskCancellationHandler {
        var finished = false
        do {
            var iterator = signal.stream.makeAsyncIterator()
            guard let fragment = await iterator.next() else {
                try await consumer.value
                throw ModelTestError.noStreamFragment
            }
            guard session.isResponding else { throw ModelTestError.completedBeforeCancellation }
            try await beforeCancellation(fragment)
            consumer.cancel()
            _ = await consumer.result
            await finishExecution()
            finished = true
            for await responding in Observations({ session.isResponding }) {
                if !responding { break }
            }
            observer.cancel()
            await observer.value
            try Task.checkCancellation()
            return responseText(in: session.transcript, after: base)
        } catch {
            consumer.cancel()
            observer.cancel()
            _ = await consumer.result
            if !finished { await finishExecution() }
            await observer.value
            throw error
        }
    } onCancel: {
        consumer.cancel()
        observer.cancel()
        signal.continuation.finish()
    }
}

actor RequestOverlapProbe {
    struct Snapshot: Sendable {
        let maximumDelegating: Int
        let events: [String]
    }

    private let onRegistered: @Sendable () -> Void
    private var arrivals = 0
    private var waiters: [AsyncStream<Void>.Continuation] = []
    private var delegating: Set<UUID> = []
    private var maximumDelegating = 0
    private var events: [String] = []

    init(onRegistered: @escaping @Sendable () -> Void = {}) {
        self.onRegistered = onRegistered
    }

    func enter(_ id: UUID) async throws {
        arrivals += 1
        events.append("enter:\(id)")
        if arrivals > 2 { return }
        let signal = AsyncStream<Void>.makeStream()
        waiters.append(signal.continuation)
        onRegistered()
        if arrivals == 2 {
            for waiter in waiters { waiter.finish() }
            waiters.removeAll()
        }
        var iterator = signal.stream.makeAsyncIterator()
        _ = await iterator.next()
        try Task.checkCancellation()
    }

    func beginDelegating(_ id: UUID) {
        delegating.insert(id)
        maximumDelegating = max(maximumDelegating, delegating.count)
        events.append("delegate:\(id)")
    }

    func finish(_ id: UUID) {
        delegating.remove(id)
        events.append("finish:\(id)")
    }

    func snapshot() -> Snapshot {
        Snapshot(maximumDelegating: maximumDelegating, events: events)
    }
}

struct ObservedLanguageModel<Base: LanguageModel>: LanguageModel {
    struct Executor: LanguageModelExecutor {
        typealias Model = ObservedLanguageModel<Base>
        private let base: Base.Executor

        init(configuration: Base.Executor.Configuration) throws {
            base = try Base.Executor(configuration: configuration)
        }

        nonisolated(nonsending) func respond(
            to request: LanguageModelExecutorGenerationRequest, model: Model,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            try await model.probe.enter(request.id)
            await model.probe.beginDelegating(request.id)
            do {
                try await base.respond(to: request, model: model.base, streamingInto: channel)
                await model.probe.finish(request.id)
            } catch {
                await model.probe.finish(request.id)
                throw error
            }
        }
    }

    let base: Base
    let probe: RequestOverlapProbe
    var capabilities: LanguageModelCapabilities { base.capabilities }
    var executorConfiguration: Base.Executor.Configuration { base.executorConfiguration }
}

final class ModelCompletionLatch: Sendable {
    private struct State {
        var value: Bool?
        var waiters: [AsyncStream<Bool>.Continuation] = []
    }

    private let state = Mutex(State())

    func finish(cancelled: Bool) {
        let waiters = state.withLock { state in
            state.value = cancelled
            let result = state.waiters
            state.waiters.removeAll()
            return result
        }
        for waiter in waiters {
            waiter.yield(cancelled)
            waiter.finish()
        }
    }

    func wait() async -> Bool? {
        let signal = AsyncStream<Bool>.makeStream()
        let completed: Bool? = state.withLock { state -> Bool? in
            if let value = state.value { return value }
            state.waiters.append(signal.continuation)
            return nil
        }
        if let completed { return completed }
        var iterator = signal.stream.makeAsyncIterator()
        return await iterator.next()
    }
}

struct StreamingProbeModel: LanguageModel {
    enum Behavior: Sendable {
        case paired(RequestOverlapProbe)
        case waitForCancellation(AsyncStream<Bool>.Continuation)
        case secondFragment(RequestOverlapProbe, ModelCompletionLatch)
    }

    struct Executor: LanguageModelExecutor {
        typealias Model = StreamingProbeModel
        init(configuration: UUID) {}

        nonisolated(nonsending) func respond(
            to request: LanguageModelExecutorGenerationRequest, model: Model,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            model.onStart()
            let finished: AsyncStream<Bool>.Continuation?
            switch model.behavior {
            case .paired: finished = nil
            case .waitForCancellation(let continuation): finished = continuation
            case .secondFragment: finished = nil
            }
            defer {
                finished?.yield(Task.isCancelled)
                finished?.finish()
                if case .secondFragment(_, let completion) = model.behavior {
                    completion.finish(cancelled: Task.isCancelled)
                }
            }
            if case .paired(let barrier) = model.behavior { try await barrier.enter(request.id) }
            await channel.send(.response(action: .appendText("fragment", tokenCount: 1)))
            if case .secondFragment(let gate, _) = model.behavior {
                try await gate.enter(request.id)
                await channel.send(.response(action: .appendText(" PELICAN", tokenCount: 1)))
            }
            if finished != nil {
                let wait = AsyncStream<Void>.makeStream()
                var iterator = wait.stream.makeAsyncIterator()
                _ = await iterator.next()
                try Task.checkCancellation()
            }
            await channel.send(.response(action: .updateMetadata([:])))
        }
    }

    let behavior: Behavior
    let executorConfiguration = UUID()
    let capabilities: LanguageModelCapabilities
    let onStart: @Sendable () -> Void

    init(
        behavior: Behavior, supportsReasoning: Bool = true,
        onStart: @escaping @Sendable () -> Void = {}
    ) {
        self.behavior = behavior
        capabilities = LanguageModelCapabilities(supportsReasoning ? [.reasoning] : [])
        self.onStart = onStart
    }
}
