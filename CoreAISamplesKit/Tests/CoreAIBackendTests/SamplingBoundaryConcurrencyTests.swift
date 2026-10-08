import Foundation
import FoundationModels
import Observation
import Synchronization
import Testing

@_spi(Verification) @testable import CoreAIBackend

extension ModelIntegrationTests {
    @Test(
        "Seeded top-k and top-p reproduce complete generated token sequences with reuse disabled",
        .enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
    func seededSamplingRepeats(path: String) async throws {
        let repetitions = try ModelTestSettings.repetitions("COREAI_SEED_REPETITIONS", default: 20)
        var settings = LocalLanguageModel.VerificationSettings()
        settings.reusesState = false
        let model = try await LocalLanguageModel(
            bundle: URL(filePath: path), contextLimit: ModelTestSettings.contextLimit, overflow: .error,
            verification: settings)
        let modes: [(String, @Sendable (UInt64) -> GenerationOptions.SamplingMode)] = [
            ("top-k", { .random(top: 40, seed: $0) }),
            ("top-p", { .random(probabilityThreshold: 0.95, seed: $0) }),
        ]
        func sample(_ mode: GenerationOptions.SamplingMode) async throws -> [Int32] {
            let session = LanguageModelSession(model: model)
            let response = try await session.respond(
                to: "Write a short four-line poem about the sea.",
                options: .init(samplingMode: mode, temperature: 0.8, maximumResponseTokens: 512),
                contextOptions: ModelTestSettings.context)
            try #require(try ModelTestSettings.metadata(session, "stopReason", as: String.self) == "stopToken")
            #expect(response.usage.input.cachedTokenCount == 0)
            #expect(
                try ModelTestSettings.metadata(session, "prefillTokens", as: Int.self)
                    == response.usage.input.totalTokenCount)
            let processed = await model.processedTokens()
            try #require(
                processed.count == response.usage.input.totalTokenCount + response.usage.output.totalTokenCount)
            let generated = Array(processed.dropFirst(response.usage.input.totalTokenCount))
            try #require(!generated.isEmpty)
            return generated
        }
        for (name, mode) in modes {
            let expected = try await sample(mode(7))
            for repetition in 1..<repetitions {
                let actual = try await sample(mode(7))
                #expect(actual == expected, "\(name) repetition \(repetition + 1) changed the token sequence")
            }
            let different = try await sample(mode(8))
            #expect(different != expected, "\(name) did not change when its seed changed")
            print("seed-check mode=\(name) repetitions=\(repetitions) reuse=false seed7=\(expected) seed8=\(different)")
        }
    }

    @Test(
        "Both overflow policies enforce all four exact 16384-token boundaries and retain valid state",
        .enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
    func contextLimitBoundaries(path: String) async throws {
        for policy in [LocalLanguageModel.OverflowPolicy.error, .dropOldToolOutputs] {
            let model = try await LocalLanguageModel(
                bundle: URL(filePath: path), contextLimit: ModelTestSettings.contextLimit, overflow: policy)
            for offset in [-2, -1, 0, 1] {
                let input = try BoundaryPrompt.make(
                    tokenizer: model.runtime.tokenizer, target: ModelTestSettings.contextLimit + offset)
                let session = LanguageModelSession(model: model)
                let before = await model.processedState()
                let options = GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8)
                if offset < 0 {
                    let response = try await session.respond(
                        to: input.text, options: options, contextOptions: ModelTestSettings.context)
                    #expect(response.usage.input.totalTokenCount == input.tokens.count)
                    #expect((1...(-offset)).contains(response.usage.output.totalTokenCount))
                    let reason = try ModelTestSettings.metadata(session, "stopReason", as: String.self)
                    let lastUnprocessed = reason == "maxTokens" ? 1 : 0
                    let processed = await model.processedTokens()
                    #expect(Array(processed.prefix(input.tokens.count)) == input.tokens)
                    #expect(
                        processed.count == input.tokens.count
                            + response.usage.output.totalTokenCount - lastUnprocessed)
                    #expect(try ModelTestSettings.metadata(session, "truncatedToolOutputs", as: Int.self) == 0)
                } else {
                    await #expect {
                        try await session.respond(
                            to: input.text, options: options, contextOptions: ModelTestSettings.context)
                    } throws: { error in
                        guard case LanguageModelError.contextSizeExceeded(let detail) = error else { return false }
                        return detail.contextSize == ModelTestSettings.contextLimit
                            && detail.tokenCount == input.tokens.count
                    }
                    let after = await model.processedState()
                    #expect(after.tokens == before.tokens)
                    #expect(after.logits == before.logits)
                }
                try await ModelTestSettings.checkState(model, label: "boundary-\(policy)-\(offset)")
            }
        }
    }

    @Test(
        "Old tool outputs shrink to reserve space and irreducible overflow preserves the previous state",
        .enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
    func toolOutputBoundary(path: String) async throws {
        let model = try await LocalLanguageModel(
            bundle: URL(filePath: path), contextLimit: ModelTestSettings.contextLimit, overflow: .dropOldToolOutputs)
        let history = try BoundaryPrompt.toolHistory()
        let input = try BoundaryPrompt.make(
            tokenizer: model.runtime.tokenizer, target: ModelTestSettings.contextLimit - 1, history: history)
        let session = LanguageModelSession(model: model, transcript: Transcript(entries: input.history))
        let response = try await session.respond(
            to: input.text, options: .init(samplingMode: .greedy, maximumResponseTokens: 8),
            contextOptions: ModelTestSettings.context)
        #expect(response.usage.input.totalTokenCount + 8 <= ModelTestSettings.contextLimit)
        #expect(response.usage.input.totalTokenCount < input.tokens.count)
        #expect(try ModelTestSettings.metadata(session, "truncatedToolOutputs", as: Int.self) == 1)
        try await ModelTestSettings.checkState(model, label: "tool-output-shrunk")

        let impossible = try BoundaryPrompt.make(
            tokenizer: model.runtime.tokenizer, target: ModelTestSettings.contextLimit + 4096, history: history)
        let rejected = LanguageModelSession(model: model, transcript: Transcript(entries: impossible.history))
        let before = await model.processedState()
        await #expect {
            try await rejected.respond(
                to: impossible.text, options: .init(samplingMode: .greedy, maximumResponseTokens: 8),
                contextOptions: ModelTestSettings.context)
        } throws: { error in
            guard case LanguageModelError.contextSizeExceeded(let detail) = error else { return false }
            return detail.contextSize == ModelTestSettings.contextLimit
                && detail.tokenCount >= ModelTestSettings.contextLimit
                && detail.tokenCount < impossible.tokens.count
        }
        let after = await model.processedState()
        #expect(after.tokens == before.tokens)
        #expect(after.logits == before.logits)
        try await ModelTestSettings.checkState(model, label: "tool-output-irreducible")
    }

    @Test(
        "Two sessions overlap for fifty pairs without marker leakage and recover from first-fragment cancellation",
        .enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
    func twoSessionsShareOneModel(path: String) async throws {
        let pairs = try ModelTestSettings.repetitions("COREAI_CONCURRENCY_PAIRS", default: 50)
        let reloadPolicy = try ModelTestSettings.reloadPolicy()
        let base = try await LocalLanguageModel(
            bundle: URL(filePath: path), contextLimit: ModelTestSettings.contextLimit, reloadPolicy: reloadPolicy)
        var reloads = 0
        for pair in 0..<pairs {
            let probe = RequestOverlapProbe()
            let model = ObservedLanguageModel(base: base, probe: probe)
            let firstMarker = "QXZV9274RMT6"
            let secondMarker = "NBKF5831WDS9"
            let first = LanguageModelSession(model: model, instructions: "Your private marker is \(firstMarker).")
            let second = LanguageModelSession(model: model, instructions: "Your private marker is \(secondMarker).")
            let cancelling = pair.isMultiple(of: 5)
            if cancelling { first.transcriptErrorHandlingPolicy = .preserveTranscript }
            async let firstResult: String = {
                if cancelling {
                    return try await cancelAfterFirstFragment(
                        first, prompt: "Begin with your private marker, then tell a long story about the ocean.",
                        maximumTokens: 256, waitUntilFinished: { _ = await base.processedTokens() })
                }
                return try await first.respond(
                    to: "Write your private marker, then describe the ocean in three sentences.",
                    options: .init(samplingMode: .greedy, maximumResponseTokens: 96),
                    contextOptions: ModelTestSettings.context
                ).content
            }()
            async let secondResult = second.respond(
                to: "Write your private marker, then describe a forest in three sentences.",
                options: .init(samplingMode: .greedy, maximumResponseTokens: 96),
                contextOptions: ModelTestSettings.context)
            let (firstText, secondResponse) = try await (firstResult, secondResult)
            if case .response(let last) = second.transcript.last, last.metadata["reloadSeconds"] != nil { reloads += 1 }
            #expect(!firstText.isEmpty && !secondResponse.content.isEmpty)
            #expect(!firstText.uppercased().contains(secondMarker))
            #expect(!secondResponse.content.uppercased().contains(firstMarker))
            let snapshot = await probe.snapshot()
            #expect(snapshot.maximumDelegating == 2, "Pair \(pair) did not overlap at the runtime executor boundary")
            print(
                "concurrency-check pair=\(pair) cancelled=\(cancelling) maximumDelegating=\(snapshot.maximumDelegating) events=\(snapshot.events)"
            )
            if cancelling {
                let recovered = try await first.respond(
                    to: "Write your private marker and one short sentence.",
                    options: .init(samplingMode: .greedy, maximumResponseTokens: 32),
                    contextOptions: ModelTestSettings.context)
                #expect(!recovered.content.isEmpty)
                #expect(!recovered.content.uppercased().contains(secondMarker))
            }
        }
        if reloadPolicy.footprintLimitBytes > 0 { #expect(reloads > 0) }
        print("concurrency-reloads=\(reloads)")
        try await ModelTestSettings.checkState(base, label: "two-sessions-final")
    }
}

@Suite(.timeLimit(.minutes(1)))
struct ModelTestSupportTests {
    @Test(
        "Exact boundary fixtures render without loading a model",
        .enabled(if: !checkPaths.isEmpty), arguments: checkPaths)
    func boundaryTokenCounts(path: String) async throws {
        let tokenizer = try await ModelBundle(url: URL(filePath: path)).loadTokenizer()
        for target in [16_382, 16_383, 16_384, 16_385] {
            #expect(try BoundaryPrompt.make(tokenizer: tokenizer, target: target).tokens.count == target)
        }
        let history = try BoundaryPrompt.toolHistory()
        #expect(try BoundaryPrompt.make(tokenizer: tokenizer, target: 16_383, history: history).tokens.count == 16_383)
        #expect(try BoundaryPrompt.make(tokenizer: tokenizer, target: 20_480, history: history).tokens.count == 20_480)
    }

    @Test("A request cancelled while waiting at the overlap gate releases its waiter")
    func gateCancellation() async throws {
        let started = AsyncStream<Void>.makeStream()
        let probe = RequestOverlapProbe(onRegistered: {
            started.continuation.yield(())
            started.continuation.finish()
        })
        let task = Task { try await probe.enter(UUID()) }
        for await _ in started.stream { break }
        task.cancel()
        let result = await task.result
        if case .failure(let error) = result {
            #expect(error is CancellationError)
        } else {
            Issue.record("A cancelled waiter did not throw")
        }
    }
}

extension ModelTestSupportTests {
    @Test("The overlap observer records both delegated requests before either completes")
    func recordsOverlappingDelegates() async throws {
        let probe = RequestOverlapProbe()
        let model = ObservedLanguageModel(
            base: StreamingProbeModel(behavior: .paired(RequestOverlapProbe())), probe: probe)
        let first = LanguageModelSession(model: model)
        let second = LanguageModelSession(model: model)
        async let a = first.respond(to: "First")
        async let b = second.respond(to: "Second")
        let responses = try await (a, b)
        #expect(responses.0.content == "fragment" && responses.1.content == "fragment")
        let snapshot = await probe.snapshot()
        #expect(snapshot.maximumDelegating == 2)
        #expect(snapshot.events.filter { $0.hasPrefix("delegate:") }.count == 2)
        #expect(snapshot.events.filter { $0.hasPrefix("finish:") }.count == 2)
        let finish = try #require(snapshot.events.firstIndex { $0.hasPrefix("finish:") })
        #expect(snapshot.events[..<finish].filter { $0.hasPrefix("delegate:") }.count == 2)
    }

    @Test("Cancellation waits for a real stream fragment and awaits the cancelled operation")
    func cancellationUsesFragment() async throws {
        let finished = AsyncStream<Bool>.makeStream()
        let session = LanguageModelSession(
            model: StreamingProbeModel(behavior: .waitForCancellation(finished.continuation)))
        session.transcriptErrorHandlingPolicy = .preserveTranscript
        let fragment = try await cancelAfterFirstFragment(
            session, prompt: "Start streaming", maximumTokens: 32,
            waitUntilFinished: {
                var iterator = finished.stream.makeAsyncIterator()
                #expect(await iterator.next() == true)
            })
        #expect(fragment == "fragment")
        #expect(!session.isResponding)
    }

    @Test("Cancellation racing with completion inspects the complete retained response")
    func cancellationCollectsFinalPartial() async throws {
        let gate = RequestOverlapProbe()
        let completed = ModelCompletionLatch()
        let session = LanguageModelSession(
            model: StreamingProbeModel(behavior: .secondFragment(gate, completed)))
        session.transcriptErrorHandlingPolicy = .preserveTranscript
        let start = session.transcript.count
        let result = try await cancelAfterFirstFragment(
            session, prompt: "Emit two controlled fragments", maximumTokens: 32,
            beforeCancellation: { first in
                #expect(first == "fragment")
                try await gate.enter(UUID())
                #expect(await completed.wait() == false)
                for await responding in Observations({ session.isResponding }) {
                    if !responding { return }
                }
                throw CancellationError()
            },
            waitUntilFinished: { #expect(await completed.wait() == false) })
        #expect(result == "fragment PELICAN")
        #expect(result == responseText(in: session.transcript, after: start))
        #expect(!session.isResponding)
    }

    @Test("A rejected request does not wait for an executor that never started")
    func cancellationRejectsBeforeExecution() async throws {
        let state = Mutex((started: false, waited: false))
        let finished = AsyncStream<Bool>.makeStream()
        let session = LanguageModelSession(
            model: StreamingProbeModel(
                behavior: .waitForCancellation(finished.continuation), supportsReasoning: false,
                onStart: { state.withLock { $0.started = true } }))
        await #expect {
            try await cancelAfterFirstFragment(
                session, prompt: "Reject unsupported reasoning", maximumTokens: 32,
                executionStarted: { state.withLock { $0.started } },
                waitUntilFinished: {
                    state.withLock { $0.waited = true }
                    var iterator = finished.stream.makeAsyncIterator()
                    _ = await iterator.next()
                })
        } throws: { error in
            guard case LanguageModelError.unsupportedCapability(let detail) = error else { return false }
            return detail.capability == .reasoning
        }
        #expect(!state.withLock { $0.started || $0.waited })
        #expect(!session.isResponding)
        finished.continuation.finish()
    }

    @Test("Parent cancellation joins the executor after cancelling the response consumer")
    func parentCancellationJoinsExecution() async throws {
        let entered = AsyncStream<Void>.makeStream()
        let finished = AsyncStream<Bool>.makeStream()
        let state = Mutex((started: false, joined: false))
        let session = LanguageModelSession(
            model: StreamingProbeModel(
                behavior: .waitForCancellation(finished.continuation),
                onStart: { state.withLock { $0.started = true } }))
        session.transcriptErrorHandlingPolicy = .preserveTranscript
        let operation = Task {
            try await cancelAfterFirstFragment(
                session, prompt: "Wait for parent cancellation", maximumTokens: 32,
                beforeCancellation: { _ in
                    entered.continuation.yield(())
                    entered.continuation.finish()
                    let wait = AsyncStream<Void>.makeStream()
                    var iterator = wait.stream.makeAsyncIterator()
                    _ = await iterator.next()
                    try Task.checkCancellation()
                },
                executionStarted: { state.withLock { $0.started } },
                waitUntilFinished: {
                    var iterator = finished.stream.makeAsyncIterator()
                    let cancelled = await iterator.next()
                    state.withLock { $0.joined = cancelled == true }
                })
        }
        for await _ in entered.stream { break }
        operation.cancel()
        if case .failure(let error) = await operation.result {
            #expect(error is CancellationError)
        } else {
            Issue.record("Parent cancellation did not end the helper")
        }
        #expect(state.withLock { $0.started && $0.joined })
        #expect(!session.isResponding)
    }

}
