import Foundation
import FoundationModels
import Observation
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

func cancelAfterFirstFragment(
    _ session: LanguageModelSession, prompt: String, maximumTokens: Int
) async throws -> String {
    let base = session.transcript.count
    let signal = AsyncStream<String>.makeStream(bufferingPolicy: .bufferingOldest(1))
    let observer = Task {
        for await transcript in Observations({ session.transcript }) {
            let text = transcript.dropFirst(base).compactMap { entry -> String? in
                guard case .response(let response) = entry else { return nil }
                return response.segments.compactMap { segment -> String? in
                    if case .text(let text) = segment { return text.content }
                    return nil
                }.joined()
            }.joined()
            if !text.isEmpty {
                signal.continuation.yield(text)
                return
            }
        }
    }
    let consumer = Task {
        defer { signal.continuation.finish() }
        for try await _ in session.streamResponse(
            to: prompt, options: .init(samplingMode: .greedy, maximumResponseTokens: maximumTokens),
            contextOptions: ModelTestSettings.context)
        {}
    }
    return try await withTaskCancellationHandler {
        do {
            var iterator = signal.stream.makeAsyncIterator()
            guard let fragment = await iterator.next() else {
                try await consumer.value
                throw ModelTestError.noStreamFragment
            }
            guard session.isResponding else { throw ModelTestError.completedBeforeCancellation }
            consumer.cancel()
            _ = await consumer.result
            for await responding in Observations({ session.isResponding }) {
                if !responding { break }
            }
            observer.cancel()
            await observer.value
            try Task.checkCancellation()
            return fragment
        } catch {
            consumer.cancel()
            observer.cancel()
            _ = await consumer.result
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

    private var arrivals = 0
    private var waiters: [AsyncStream<Void>.Continuation] = []
    private var delegating: Set<UUID> = []
    private var maximumDelegating = 0
    private var events: [String] = []

    func enter(_ id: UUID) async throws {
        arrivals += 1
        events.append("enter:\(id)")
        if arrivals > 2 { return }
        let signal = AsyncStream<Void>.makeStream()
        waiters.append(signal.continuation)
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

struct StreamingProbeModel: LanguageModel {
    enum Behavior: Sendable {
        case paired(RequestOverlapProbe)
        case waitForCancellation(AsyncStream<Bool>.Continuation)
    }

    struct Executor: LanguageModelExecutor {
        typealias Model = StreamingProbeModel
        init(configuration: UUID) {}

        nonisolated(nonsending) func respond(
            to request: LanguageModelExecutorGenerationRequest, model: Model,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            if case .paired(let barrier) = model.behavior { try await barrier.enter(request.id) }
            await channel.send(.response(action: .appendText("fragment", tokenCount: 1)))
            if case .waitForCancellation(let finished) = model.behavior {
                defer {
                    finished.yield(Task.isCancelled)
                    finished.finish()
                }
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
    let capabilities = LanguageModelCapabilities([.reasoning])
}
