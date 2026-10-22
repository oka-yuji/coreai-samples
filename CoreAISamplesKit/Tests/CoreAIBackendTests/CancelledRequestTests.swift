import Foundation
import FoundationModels
import Testing

@_spi(Verification) @testable import CoreAIBackend

extension Runtime {
    func barrier() {}
}

@Suite(.serialized, .timeLimit(.minutes(1)))
struct CancelledRequestTests {
    enum Point: Sendable {
        case send(afterSample: Int)
        case metadata(afterSample: Int)
        case usage
    }

    enum TerminalSend: String, CaseIterable, Sendable {
        case responseMetadata
        case toolCallMetadata
        case usage
    }

    static let heldTextTokenizer = #"""
        {"added_tokens": [{"id": 3, "content": "<|im_end|>", "special": true}],
         "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "\uFFFD": 2}, "merges": []}}
        """#

    static let toolCallTokenizer = #"""
        {"added_tokens": [{"id": 3, "content": "<|im_end|>", "special": true}],
         "model": {"type": "BPE", "merges": [], "vocab": {"<tool_call>": 0,
           "{\"name\": \"add\", \"arguments\": {\"a\": 1, \"b\": 2}}": 1, "</tool_call>": 2, "a": 4}}}
        """#

    static let addTool = Transcript.ToolDefinition(
        name: "add", description: "Add two integers.", parameters: AddArguments.generationSchema)

    static func fixture(_ tokens: [Int], tokenizer: String? = nil) async throws -> ReloadTestModel {
        var settings = LocalLanguageModel.VerificationSettings()
        settings.measurement = .summary
        let fixture = try await ReloadTestModel.make(
            policy: .init(readInterval: 0), limit: 1024, settings: settings, tokenizer: tokenizer)
        fixture.control.state.withLock { $0.generationTokens = tokens }
        return fixture
    }

    static func cancelInSend(
        _ fixture: ReloadTestModel, limit: Int, reading: Int, at point: Point, tools: [Transcript.ToolDefinition] = []
    ) async -> (read: Int, error: (any Error)?) {
        let channel = LanguageModelExecutorGenerationChannel()
        let hold = AsyncStream<Void>.makeStream()
        let reached = AsyncStream<Void>.makeStream()
        let reader = Task {
            var events = channel.makeAsyncIterator()
            var read = 0
            while read < reading, try await events.next() != nil { read += 1 }
            reached.continuation.yield()
            var held = hold.stream.makeAsyncIterator()
            _ = await held.next()
            return read
        }
        if case .metadata(let sample) = point { fixture.control.state.withLock { $0.signalStatsAfterSample = sample } }
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(), transcript: Transcript(entries: [.prompt(.init(segments: [.text(.init(content: "a"))]))]),
            enabledTools: tools, generationOptions: .init(samplingMode: .greedy, maximumResponseTokens: limit),
            contextOptions: .init(), metadata: [:])
        let executor = Task { try await fixture.runtime.respond(to: request, streamingInto: channel) }
        switch point {
        case .send(let sample):
            var samples = fixture.control.sampling.stream.makeAsyncIterator()
            while let count = await samples.next(), count < sample {}
        case .metadata:
            var reads = fixture.control.statsRead.stream.makeAsyncIterator()
            _ = await reads.next()
        case .usage:
            var arrived = reached.stream.makeAsyncIterator()
            _ = await arrived.next()
        }
        await fixture.runtime.barrier()
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
        let cancelled = await Self.cancelInSend(fixture, limit: limit, reading: 2, at: .send(afterSample: 3))
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
        let cancelled = await Self.cancelInSend(fixture, limit: 8, reading: 0, at: .send(afterSample: 2))
        #expect(cancelled.read == 0)
        #expect(cancelled.error is CancellationError, "\(String(describing: cancelled.error))")
        #expect(fixture.control.state.withLock { $0.steps } == 1)
        #expect(await fixture.model.processedTokens() == [0, 2])
        #expect(await fixture.model.lastGeneratedTokens() == [2, 1])
    }

    @Test("A request cancelled in a terminal send after EOS does not end normally or keep its generation for reuse")
    func cancelInTerminalSend() async throws {
        let fixture = try await Self.fixture([0, 1, 3])
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        let cancelled = await Self.cancelInSend(fixture, limit: 8, reading: 2, at: .send(afterSample: 3))
        #expect(cancelled.error is CancellationError, "\(String(describing: cancelled.error))")
        #expect(await fixture.model.lastRequestMeasurement()?.outcome == .cancelled)
        let session = LanguageModelSession(
            model: fixture.model,
            transcript: Transcript(entries: [
                .prompt(.init(segments: [.text(.init(content: "a"))])),
                .response(.init(segments: [.text(.init(content: "ab"))])),
            ]))
        _ = try await session.respond(to: "a", options: .init(samplingMode: .greedy, maximumResponseTokens: 1))
        #expect(try ModelTestSettings.metadata(session, "splicedGenerations", as: Int.self) == 0)
        let held = try await Self.fixture([2, 3], tokenizer: Self.heldTextTokenizer)
        defer { try? FileManager.default.removeItem(at: held.folder) }
        let drained = await Self.cancelInSend(held, limit: 8, reading: 0, at: .send(afterSample: 2))
        #expect(drained.read == 0)
        #expect(drained.error is CancellationError, "\(String(describing: drained.error))")
        #expect(await held.model.lastRequestMeasurement()?.outcome == .cancelled)
    }

    @Test(
        "A request cancelled inside its metadata or usage send ends with CancellationError and keeps no generation",
        arguments: TerminalSend.allCases)
    func cancelInEachTerminalSend(_ send: TerminalSend) async throws {
        let calling = send == .toolCallMetadata
        let fixture = try await Self.fixture(
            calling ? [0, 1, 2, 3] : [0, 1, 3], tokenizer: calling ? Self.toolCallTokenizer : nil)
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        let cancelled =
            switch send {
            case .responseMetadata:
                await Self.cancelInSend(fixture, limit: 8, reading: 2, at: .metadata(afterSample: 3))
            case .toolCallMetadata:
                await Self.cancelInSend(
                    fixture, limit: 8, reading: 1, at: .metadata(afterSample: 4), tools: [Self.addTool])
            case .usage: await Self.cancelInSend(fixture, limit: 8, reading: 3, at: .usage)
            }
        #expect(cancelled.read == (send == .usage ? 3 : calling ? 1 : 2))
        #expect(cancelled.error is CancellationError, "\(String(describing: cancelled.error))")
        #expect(await fixture.model.lastRequestMeasurement()?.outcome == .cancelled)
        guard !calling else { return }
        let session = LanguageModelSession(
            model: fixture.model,
            transcript: Transcript(entries: [
                .prompt(.init(segments: [.text(.init(content: "a"))])),
                .response(.init(segments: [.text(.init(content: "ab"))])),
            ]))
        _ = try await session.respond(to: "a", options: .init(samplingMode: .greedy, maximumResponseTokens: 1))
        #expect(try ModelTestSettings.metadata(session, "splicedGenerations", as: Int.self) == 0)
    }

    @Test(
        "A session preserving its transcript answers the next request right after a cancelled response",
        arguments: 0..<30)
    func nextRequestAfterCancellationPreservingTranscript(run _: Int) async throws {
        let fixture = try await Self.fixture([1])
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        let session = LanguageModelSession(model: fixture.model)
        session.transcriptErrorHandlingPolicy = .preserveTranscript
        fixture.control.state.withLock { $0.blockAtStep = $0.steps + 1 }
        _ = try await withTaskCancellationHandler {
            try await cancelAfterFirstFragment(
                session, prompt: "aaaa", maximumTokens: 8, context: .init(),
                beforeCancellation: { _ in
                    var blocked = fixture.control.stepping.stream.makeAsyncIterator()
                    _ = await blocked.next()
                },
                waitUntilFinished: { _ = await fixture.model.processedTokens() })
        } onCancel: {
            fixture.control.resumeStep()
        }
        try #require(await fixture.model.lastRequestMeasurement()?.outcome == .cancelled)
        let next = try await session.respond(
            to: "aaaa", options: .init(samplingMode: .greedy, maximumResponseTokens: 8))
        #expect(next.content == "bbbbbbbb")
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
            let sharedTokens = try ModelTestSettings.metadata(session, "sharedTokens", as: Int.self)
            let shared = zip(before, processed.prefix(input.totalTokenCount)).prefix { $0 == $1 }.count
            #expect(try ModelTestSettings.metadata(session, "stopReason", as: String.self) == "stopToken")
            #expect(session.transcript.filter { if case .prompt = $0 { true } else { false } }.count == 1)
            #expect(processed.count == input.totalTokenCount + after.usage.output.totalTokenCount)
            #expect(shared == sharedTokens, "run \(run): common prefix \(shared), sharedTokens \(sharedTokens)")
            #expect(input.cachedTokenCount <= sharedTokens)
            #expect(input.cachedTokenCount + prefilled == input.totalTokenCount)
            #expect(!after.content.isEmpty)
            print(
                "cancel-next-model run=\(run) left=\(left.count) before=\(before.count) prompt=\(input.totalTokenCount)"
                    + " shared=\(shared) sharedTokens=\(sharedTokens) cached=\(input.cachedTokenCount)"
                    + " prefilled=\(prefilled) answer=\(after.content)")
            try await ModelTestSettings.checkState(model, label: "cancel-next-\(run)")
        }
    }
}
