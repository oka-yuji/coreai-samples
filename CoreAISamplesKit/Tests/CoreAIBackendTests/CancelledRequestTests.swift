import Foundation
import FoundationModels
import Testing

@_spi(Verification) @testable import CoreAIBackend

@Suite(.serialized, .timeLimit(.minutes(1)))
struct CancelledRequestTests {
    enum Condition: String, CaseIterable, Sendable {
        case firstStep
        case thirdStep
        case freeRunning
        case firstStepAfterSendCancellation
        case thirdStepAfterSendCancellation
        case freeRunningAfterSendCancellation
        case firstStepPreservingTranscript

        var blockedStep: Int? {
            switch self {
            case .firstStep, .firstStepAfterSendCancellation, .firstStepPreservingTranscript: 1
            case .thirdStep, .thirdStepAfterSendCancellation: 3
            case .freeRunning, .freeRunningAfterSendCancellation: nil
            }
        }

        var followsSendCancellation: Bool { rawValue.hasSuffix("AfterSendCancellation") }

        static let requested: [Condition] = {
            let value = ProcessInfo.processInfo.environment["COREAI_CANCEL_NEXT_CONDITIONS"] ?? ""
            let named = value.split(separator: ",").compactMap { Condition(rawValue: String($0)) }
            return value == "all" ? allCases : named
        }()
    }

    static let heldTextTokenizer = #"""
        {"added_tokens": [{"id": 3, "content": "<|im_end|>", "special": true}],
         "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "\uFFFD": 2}, "merges": []}}
        """#

    static func fixture(_ tokens: [Int], tokenizer: String? = nil) async throws -> ReloadTestModel {
        var settings = LocalLanguageModel.VerificationSettings()
        settings.measurement = .summary
        let fixture = try await ReloadTestModel.make(
            policy: .init(readInterval: 0), limit: 1024, settings: settings, tokenizer: tokenizer)
        fixture.control.state.withLock { $0.generationTokens = tokens }
        return fixture
    }

    static func cancelInSend(_ fixture: ReloadTestModel, limit: Int, reading: Int, sample: Int) async
        -> (read: Int, error: (any Error)?)
    {
        let channel = LanguageModelExecutorGenerationChannel()
        let hold = AsyncStream<Void>.makeStream()
        let reader = Task {
            var events = channel.makeAsyncIterator()
            var read = 0
            while read < reading, try await events.next() != nil { read += 1 }
            var held = hold.stream.makeAsyncIterator()
            _ = await held.next()
            return read
        }
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(), transcript: Transcript(entries: [.prompt(.init(segments: [.text(.init(content: "a"))]))]),
            enabledTools: [], generationOptions: .init(samplingMode: .greedy, maximumResponseTokens: limit),
            contextOptions: .init(), metadata: [:])
        let executor = Task { try await fixture.runtime.respond(to: request, streamingInto: channel) }
        var samples = fixture.control.sampling.stream.makeAsyncIterator()
        while let count = await samples.next(), count < sample {}
        executor.cancel()
        let result = await executor.result
        hold.continuation.finish()
        reader.cancel()
        let read = (try? await reader.value) ?? -1
        guard case .failure(let error) = result else { return (read, nil) }
        return (read, error)
    }

    @Test(
        "A request cancelled inside send neither steps the undelivered token nor keeps its generation for reuse",
        arguments: [8, 3])
    func cancelInSend(limit: Int) async throws {
        let fixture = try await Self.fixture([0, 1])
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        let cancelled = await Self.cancelInSend(fixture, limit: limit, reading: 2, sample: 3)
        #expect(cancelled.read == 2)
        #expect(cancelled.error is CancellationError, "\(String(describing: cancelled.error))")
        #expect(fixture.control.state.withLock { $0.steps } == 2)
        #expect(await fixture.model.processedTokens() == [0, 0, 1])
        #expect(await fixture.model.lastGeneratedTokens() == [0, 1, 1])
        let measured = try #require(await fixture.model.lastRequestMeasurement())
        #expect(measured.outcome == .cancelled && measured.sampledTokens == 3 && measured.steps == 2)
        let session = LanguageModelSession(
            model: fixture.model,
            transcript: Transcript(entries: [
                .prompt(.init(segments: [.text(.init(content: "a"))])),
                .response(.init(segments: [.text(.init(content: "abb"))])),
            ]))
        let next = try await session.respond(to: "a", options: .init(samplingMode: .greedy, maximumResponseTokens: 1))
        #expect(try ModelTestSettings.metadata(session, "splicedGenerations", as: Int.self) == 0)
        #expect(try ModelTestSettings.metadata(session, "sharedTokens", as: Int.self) == 1)
        #expect(next.usage.input.cachedTokenCount == 1 && next.usage.input.totalTokenCount == 4)
        #expect(await fixture.model.processedTokens() == [0, 2, 1, 0])
    }

    @Test("A request cancelled while sending text completed after a held UTF-8 token steps only the held token")
    func cancelInSendAfterHeldText() async throws {
        let fixture = try await Self.fixture([2, 1], tokenizer: Self.heldTextTokenizer)
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        let cancelled = await Self.cancelInSend(fixture, limit: 8, reading: 0, sample: 2)
        #expect(cancelled.read == 0)
        #expect(cancelled.error is CancellationError, "\(String(describing: cancelled.error))")
        #expect(fixture.control.state.withLock { $0.steps } == 1)
        #expect(await fixture.model.processedTokens() == [0, 2])
        #expect(await fixture.model.lastGeneratedTokens() == [2, 1])
    }

    @Test(
        "The same session answers the next request right after a cancelled response",
        .enabled(if: !Condition.requested.isEmpty), arguments: Condition.requested, 0..<30)
    func nextRequestAfterCancellation(condition: Condition, run: Int) async throws {
        let fixture = try await Self.fixture([1])
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        if condition.followsSendCancellation {
            let prelude = await Self.cancelInSend(fixture, limit: 8, reading: 2, sample: 3)
            try #require(prelude.error is CancellationError)
        }
        let session = LanguageModelSession(model: fixture.model)
        if condition == .firstStepPreservingTranscript { session.transcriptErrorHandlingPolicy = .preserveTranscript }
        if let step = condition.blockedStep { fixture.control.state.withLock { $0.blockAtStep = $0.steps + step } }
        let left: String
        do {
            left = try await withTaskCancellationHandler {
                try await cancelAfterFirstFragment(
                    session, prompt: "aaaa", maximumTokens: condition.blockedStep == nil ? 1000 : 8, context: .init(),
                    beforeCancellation: { _ in
                        guard condition.blockedStep != nil else { return }
                        var blocked = fixture.control.stepping.stream.makeAsyncIterator()
                        _ = await blocked.next()
                    },
                    waitUntilFinished: { _ = await fixture.model.processedTokens() })
            } onCancel: {
                fixture.control.resumeStep()
            }
        } catch {
            Issue.record("\(condition) run \(run): the first request was not cancelled while responding: \(error)")
            return
        }
        var outcome = "answered"
        var prompt = -1
        do {
            let next = try await session.respond(
                to: "aaaa", options: .init(samplingMode: .greedy, maximumResponseTokens: 8))
            prompt = next.usage.input.totalTokenCount
            if next.content.isEmpty { outcome = "empty" }
            #expect(next.content == "bbbbbbbb", "\(condition) run \(run): the next response was \(next.content)")
        } catch {
            outcome = "\(error)"
            Issue.record("\(condition) run \(run): the next request failed: \(error)")
        }
        print(
            "cancel-next condition=\(condition) run=\(run) left=\(left.count) prompt=\(prompt)"
                + " entries=\(session.transcript.count) outcome=\(outcome)")
    }
}

extension ModelIntegrationTests {
    @Test(
        "Thirty cancellations at the first fragment leave each session answering from the reusable shared prefix",
        .enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
    func sameSessionAnswersAfterCancellation(path: String) async throws {
        let model = try await LocalLanguageModel(
            bundle: URL(filePath: path), contextLimit: ModelTestSettings.contextLimit,
            reloadPolicy: .init(readInterval: 0, footprintLimitBytes: 0))
        for run in 0..<30 {
            let session = LanguageModelSession(model: model)
            let left = try await cancelAfterFirstFragment(
                session, prompt: "Write a long story about a dragon.", maximumTokens: 2048,
                waitUntilFinished: { _ = await model.processedTokens() })
            let before = await model.processedTokens()
            let after = try await session.respond(
                to: "What is the capital of Japan? Answer with one word.",
                options: GenerationOptions(maximumResponseTokens: 512), contextOptions: ModelTestSettings.context)
            let processed = await model.processedTokens()
            let input = after.usage.input
            let prefilled = try ModelTestSettings.metadata(session, "prefillTokens", as: Int.self)
            let shared = zip(before, processed.prefix(input.totalTokenCount)).prefix { $0 == $1 }.count
            #expect(try ModelTestSettings.metadata(session, "stopReason", as: String.self) == "stopToken")
            #expect(session.transcript.filter { if case .prompt = $0 { true } else { false } }.count == 1)
            #expect(processed.count == input.totalTokenCount + after.usage.output.totalTokenCount)
            #expect(input.cachedTokenCount + prefilled == input.totalTokenCount)
            #expect(shared == input.cachedTokenCount, "run \(run): shared \(shared), cached \(input.cachedTokenCount)")
            #expect(!after.content.isEmpty)
            print(
                "cancel-next-model run=\(run) left=\(left.count) before=\(before.count) prompt=\(input.totalTokenCount)"
                    + " cached=\(input.cachedTokenCount) prefilled=\(prefilled) shared=\(shared)"
                    + " answer=\(after.content)")
            try await ModelTestSettings.checkState(model, label: "cancel-next-\(run)")
        }
    }
}
