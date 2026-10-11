import Foundation
import FoundationModels
import Testing

@_spi(Verification) @testable import CoreAIBackend

@Suite(.timeLimit(.minutes(1)))
struct RuntimeReloadTests {
    @Test("Releasing a load before registration lets the factory finish without another release")
    func releaseBeforeLoadRegistration() async throws {
        let folder = try tinyBundle()
        defer { try? FileManager.default.removeItem(at: folder) }
        let bundle = try ModelBundle(url: folder)
        let control = ReloadTestControl()
        control.state.withLock { $0.blockNextLoad = true }
        control.resumeLoad()
        try await withTaskCancellationHandler {
            try await withThrowingTaskGroup(of: Bool.self) { group in
                defer {
                    control.resumeLoad()
                    control.loading.continuation.finish()
                }
                group.addTask {
                    var loading = control.loading.stream.makeAsyncIterator()
                    _ = await loading.next()
                    return false
                }
                group.addTask {
                    _ = try await control.make(bundle, 128)
                    return true
                }
                let completedWithoutAnotherRelease = try await group.next()
                #expect(completedWithoutAnotherRelease == true)
                control.resumeLoad()
                control.loading.continuation.finish()
                try await group.waitForAll()
            }
        } onCancel: {
            control.resumeLoad()
            control.loading.continuation.finish()
        }
        #expect(control.loads == 1)
    }

    @Test("A reload releases the old engine and all saved checkpoints before calling the loader")
    func releasesResourcesBeforeLoading() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 1))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        _ = try await fixture.request()
        #expect(fixture.control.loads == 1)
        #expect(fixture.control.state.withLock { !$0.liveCheckpoints.isEmpty })
        _ = try await fixture.request("b")
        #expect(fixture.control.loads == 2)
        let events = fixture.control.events
        let released = try #require(events.firstIndex(of: "released-engine:1"))
        let loaded = try #require(events.firstIndex(of: "load:2"))
        #expect(released < loaded)
        #expect(events[..<loaded].contains { $0.hasPrefix("released-checkpoint:") })
    }

    @Test("A load failure leaves empty public state and the next request retries without discarding CPU prompt history")
    func retriesAfterLoadFailure() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 1))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        _ = try await fixture.request()
        fixture.control.state.withLock { $0.failNextLoad = true }
        await #expect(throws: ReloadTestFailure.self) { try await fixture.request("b") }
        #expect(await fixture.model.processedTokens().isEmpty)
        let state = await fixture.model.processedState()
        #expect(state.tokens.isEmpty && state.logits.isEmpty)
        let response = try await fixture.request("a")
        #expect(response.content == "b")
        #expect(fixture.control.loads == 3)
        #expect(response.usage.input.cachedTokenCount == 0)
    }

    @Test("Read failures count toward the interval while validation and context fitting failures do not")
    func countsOnlyStartedReads() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 2), limit: 8)
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        _ = try await fixture.request()
        let invalid = LanguageModelSession(model: fixture.model)
        await #expect(throws: LanguageModelError.self) {
            try await invalid.respond(to: "a", contextOptions: .init(reasoningLevel: .custom("unsupported")))
        }
        await #expect(throws: LanguageModelError.self) { try await fixture.request(String(repeating: "a", count: 16)) }
        #expect(fixture.control.loads == 1)
        fixture.control.state.withLock { $0.failNextRead = true }
        await #expect(throws: ReloadTestFailure.self) { try await fixture.request("bbb") }
        #expect(fixture.control.loads == 1)
        _ = try await fixture.request()
        #expect(fixture.control.loads == 2)
    }

    @Test("A request cancelled before it starts cannot reload or increment the read counter")
    func cancellationBeforeLoading() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 2))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        _ = try await fixture.request()
        await withTaskGroup(of: Bool.self) { group in
            group.cancelAll()
            group.addTask {
                do {
                    try await fixture.runtime.respond(
                        to: directReloadRequest(
                            Transcript(entries: [.prompt(.init(segments: [.text(.init(content: "b"))]))])),
                        streamingInto: .init())
                    return false
                } catch { return true }
            }
            #expect(await group.next() == true)
        }
        _ = try await fixture.request("bb")
        #expect(fixture.control.loads == 1)
        _ = try await fixture.request("bbb")
        #expect(fixture.control.loads == 2)
    }

    @Test(
        "Cancellation during a blocked reload waits for creation and leaves the new engine available", arguments: 0..<10
    )
    func cancellationDuringLoading(iteration _: Int) async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 1))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        _ = try await fixture.request()
        fixture.control.state.withLock { $0.blockNextLoad = true }
        try await withTaskCancellationHandler {
            try await withThrowingTaskGroup(of: Bool.self) { group in
                defer { fixture.control.resumeLoad() }
                group.addTask {
                    do {
                        try await fixture.runtime.respond(
                            to: directReloadRequest(
                                Transcript(entries: [.prompt(.init(segments: [.text(.init(content: "b"))]))])),
                            streamingInto: .init())
                        return false
                    } catch { return true }
                }
                var loading = fixture.control.loading.stream.makeAsyncIterator()
                try #require(await loading.next() == 2)
                async let state = fixture.model.processedState()
                group.cancelAll()
                fixture.control.resumeLoad()
                #expect(try await group.next() == true)
                let loaded = await state
                #expect(loaded.tokens.isEmpty && loaded.logits.isEmpty)
            }
        } onCancel: {
            fixture.control.resumeLoad()
        }
        let response = try await fixture.request("bb")
        #expect(response.content == "b")
        #expect(fixture.control.loads == 2)
    }

    @Test("Requests and state readers queued during a reload see only a fully created engine")
    func waitingRequestsUseTheLoadedEngine() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 1))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        _ = try await fixture.request()
        fixture.control.state.withLock { $0.blockNextLoad = true }
        try await withTaskCancellationHandler {
            async let first = fixture.request("b")
            var loading = fixture.control.loading.stream.makeAsyncIterator()
            try #require(await loading.next() == 2)
            async let state = fixture.model.processedState()
            let continued = LanguageModelSession(
                model: fixture.model,
                transcript: Transcript(entries: [
                    .prompt(.init(id: "history", segments: [.text(.init(content: "a"))])),
                    .response(.init(segments: [.text(.init(content: "b"))])),
                ]))
            async let second = continued.respond(
                to: "a", options: .init(samplingMode: .greedy, maximumResponseTokens: 8))
            fixture.control.resumeLoad()
            let values = try await (first, second, state)
            #expect(values.0.content == "b" && values.1.content == "b")
            #expect(!values.2.tokens.isEmpty && !values.2.logits.isEmpty)
            #expect(fixture.control.loads == 2)
        } onCancel: {
            fixture.control.resumeLoad()
        }
    }

    @Test("Reload metadata preserves logical sharing while reading the full prompt again")
    func reloadMetadata() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 0, footprintLimitBytes: 100))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        let session = LanguageModelSession(model: fixture.model)
        _ = try await session.respond(to: "a", options: .init(samplingMode: .greedy, maximumResponseTokens: 8))
        let previous = await fixture.model.processedTokens()
        fixture.control.state.withLock { $0.footprint = 100 }
        let response = try await session.respond(
            to: "b", options: .init(samplingMode: .greedy, maximumResponseTokens: 1))
        let tokens = await fixture.model.processedTokens()
        let shared = zip(previous, tokens).prefix { $0 == $1 }.count
        #expect(try ModelTestSettings.metadata(session, "sharedTokens", as: Int.self) == shared)
        #expect(
            try ModelTestSettings.metadata(session, "prefillTokens", as: Int.self)
                == response.usage.input.totalTokenCount)
        #expect(try ModelTestSettings.metadata(session, "reloadSeconds", as: Double.self) >= 0)
        #expect(response.usage.input.cachedTokenCount == 0)
        #expect(fixture.control.loads == 2)
    }
}

extension RuntimeReloadTests {
    @Test("Cancellation immediately after reloading preserves raw generated token splicing on retry", arguments: 0..<10)
    func cancelledReloadPreservesSplicing(iteration _: Int) async throws {
        let control = try await ReloadTestModel.make(policy: .init(readInterval: 0))
        let reloaded = try await ReloadTestModel.make(policy: .init(readInterval: 0, footprintLimitBytes: 100))
        defer {
            try? FileManager.default.removeItem(at: control.folder)
            try? FileManager.default.removeItem(at: reloaded.folder)
        }
        for fixture in [control, reloaded] {
            fixture.control.state.withLock { $0.generationTokens = [0, 1, 3] }
        }
        let baselineSession = LanguageModelSession(model: control.model)
        let retrySession = LanguageModelSession(model: reloaded.model)
        let options = GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8)
        let baselineFirst = try await baselineSession.respond(to: "a", options: options)
        let reloadFirst = try await retrySession.respond(to: "a", options: options)
        #expect(baselineFirst.content == "ab" && reloadFirst.content == "ab")
        #expect(await control.model.processedTokens() == [0, 0, 1])
        _ = try await baselineSession.respond(to: "a", options: .init(samplingMode: .greedy, maximumResponseTokens: 1))
        let expected = await control.model.processedTokens()
        #expect(expected == [0, 0, 1, 0])
        reloaded.control.state.withLock {
            $0.footprint = 100
            $0.blockNextLoad = true
        }
        try await withTaskCancellationHandler {
            try await withThrowingTaskGroup(of: Bool.self) { group in
                defer { reloaded.control.resumeLoad() }
                group.addTask {
                    do {
                        let transcript = Transcript(
                            entries: Array(retrySession.transcript) + [
                                .prompt(.init(segments: [.text(.init(content: "a"))]))
                            ])
                        try await reloaded.runtime.respond(to: directReloadRequest(transcript), streamingInto: .init())
                        return false
                    } catch { return true }
                }
                var loading = reloaded.control.loading.stream.makeAsyncIterator()
                try #require(await loading.next() == 2)
                reloaded.control.state.withLock { $0.footprint = 0 }
                group.cancelAll()
                reloaded.control.resumeLoad()
                #expect(try await group.next() == true)
            }
        } onCancel: {
            reloaded.control.resumeLoad()
        }
        _ = try await retrySession.respond(to: "a", options: .init(samplingMode: .greedy, maximumResponseTokens: 1))
        let actual = await reloaded.model.processedTokens()
        #expect(actual == expected)
        #expect(try ModelTestSettings.metadata(retrySession, "sharedTokens", as: Int.self) == 3)
        #expect(try ModelTestSettings.metadata(retrySession, "prefillTokens", as: Int.self) == expected.count)
        #expect(reloaded.control.loads == 2)
        print("cancelled-reload-splice expected=\(expected) actual=\(actual)")
    }
}

extension RuntimeReloadTests {
    @Test("A checkpoint hit without prefill does not increment the read interval")
    func checkpointHitDoesNotCountAsARead() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 2))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        _ = try await fixture.request("a")
        let repeated = try await fixture.request("a")
        #expect(repeated.usage.input.cachedTokenCount == repeated.usage.input.totalTokenCount)
        _ = try await fixture.request("bbb")
        #expect(fixture.control.loads == 1)
        _ = try await fixture.request("bbbb")
        #expect(fixture.control.loads == 2)
    }

    @Test("A request cancelled after entering prefill counts toward the next new-conversation reload")
    func cancelledReadCounts() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 1))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        fixture.control.state.withLock { $0.blockNextRead = true }
        try await withTaskCancellationHandler {
            try await withThrowingTaskGroup(of: Bool.self) { group in
                defer { fixture.control.resumeRead() }
                group.addTask {
                    do {
                        try await fixture.runtime.respond(
                            to: directReloadRequest(
                                Transcript(entries: [.prompt(.init(segments: [.text(.init(content: "a"))]))])),
                            streamingInto: .init())
                        return false
                    } catch { return true }
                }
                var reading = fixture.control.reading.stream.makeAsyncIterator()
                try #require(await reading.next() == 1)
                group.cancelAll()
                fixture.control.resumeRead()
                #expect(try await group.next() == true)
            }
        } onCancel: {
            fixture.control.resumeRead()
        }
        #expect(await fixture.model.processedTokens().isEmpty)
        _ = try await fixture.request("b")
        #expect(fixture.control.loads == 2)
    }
}

extension RuntimeReloadTests {
    @Test("The verification token snapshot includes the last sampled token omitted from a token-limited engine state")
    func generatedTokenSnapshotIncludesTheLastToken() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 0))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        fixture.control.state.withLock { $0.generationTokens = [0, 1, 3] }
        let full = try await fixture.request("a")
        #expect(full.content == "ab")
        #expect(await fixture.model.lastGeneratedTokens() == [0, 1])
        let limited = LanguageModelSession(model: fixture.model)
        let response = try await limited.respond(
            to: "b", options: .init(samplingMode: .greedy, maximumResponseTokens: 1))
        #expect(response.usage.output.totalTokenCount == 1)
        #expect(await fixture.model.lastGeneratedTokens() == [0])
        #expect(await fixture.model.processedTokens().count == response.usage.input.totalTokenCount)
    }
}

extension RuntimeReloadTests {
    @Test(
        "Request-only rejection preserves the previous snapshot while a failed decode keeps only its partial generation"
    )
    func rejectedRequestPreservesGenerationAndPartialFailure() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 0))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        fixture.control.state.withLock { $0.generationTokens = [0, 1, 3] }
        _ = try await fixture.request()
        #expect(await fixture.model.lastGeneratedTokens() == [0, 1])
        let previousState = await fixture.model.processedState()
        let previousEvents = fixture.control.events
        let invalid = LanguageModelExecutorGenerationRequest(
            id: UUID(), transcript: Transcript(entries: [.prompt(.init(segments: [.text(.init(content: "a"))]))]),
            enabledTools: [], generationOptions: .init(), contextOptions: .init(), metadata: ["unsupported": true])
        await #expect(throws: LanguageModelError.self) {
            try await fixture.runtime.respond(to: invalid, streamingInto: .init())
        }
        #expect(await fixture.model.lastGeneratedTokens() == [0, 1])
        let rejectedState = await fixture.model.processedState()
        #expect(rejectedState.tokens == previousState.tokens && rejectedState.logits == previousState.logits)
        #expect(fixture.control.events == previousEvents)
        fixture.control.state.withLock { $0.failNextStep = true }
        await #expect(throws: ReloadTestFailure.self) { try await fixture.request("b") }
        #expect(await fixture.model.lastGeneratedTokens() == [0])
        #expect(await fixture.model.processedTokens().isEmpty)
    }

    @Test("A cancelled decode snapshot contains only its sampled tokens and no older successful result")
    func cancelledGenerationSnapshot() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 0))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        fixture.control.state.withLock { $0.generationTokens = [0, 1, 3] }
        _ = try await fixture.request()
        fixture.control.state.withLock { $0.blockNextStep = true }
        let session = LanguageModelSession(model: fixture.model)
        try await withTaskCancellationHandler {
            _ = try await cancelAfterFirstFragment(
                session, prompt: "b", maximumTokens: 8, context: .init(),
                waitUntilFinished: { _ = await fixture.model.processedTokens() })
        } onCancel: {
            fixture.control.resumeStep()
        }
        #expect(await fixture.model.lastGeneratedTokens() == [0])
        #expect(await fixture.model.processedTokens().isEmpty)
    }
}

extension RuntimeReloadTests {
    @Test("Request-only rejection neither registers a new conversation nor disturbs the existing one")
    func rejectionPreservesConversation() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 1))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        let session = LanguageModelSession(model: fixture.model)
        _ = try await session.respond(to: "a", options: .init(samplingMode: .greedy, maximumResponseTokens: 8))
        let nextConversation = Transcript(entries: [.prompt(.init(segments: [.text(.init(content: "b"))]))])
        let invalid = LanguageModelExecutorGenerationRequest(
            id: UUID(), transcript: nextConversation,
            enabledTools: [], generationOptions: .init(), contextOptions: .init(), metadata: ["unsupported": true])
        await #expect(throws: LanguageModelError.self) {
            try await fixture.runtime.respond(to: invalid, streamingInto: .init())
        }
        _ = try await session.respond(to: "b", options: .init(samplingMode: .greedy, maximumResponseTokens: 8))
        #expect(fixture.control.loads == 1)
        fixture.control.state.withLock {
            $0.failNextLoad = true
            $0.failNextRead = true
        }
        do {
            try await fixture.runtime.respond(to: directReloadRequest(nextConversation), streamingInto: .init())
            Issue.record("The accepted new conversation did not attempt a reload")
        } catch ReloadTestFailure.load {
            #expect(fixture.control.loads == 2)
        } catch {
            Issue.record("The rejected request registered its conversation: \(error)")
        }
    }

    @Test("An invalid request is rejected while another request holds the execution lock during a read")
    func rejectionDoesNotWaitForExecutionLock() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 0))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        fixture.control.state.withLock { $0.blockNextRead = true }
        try await withTaskCancellationHandler {
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask { _ = try await fixture.request() }
                do {
                    var reading = fixture.control.reading.stream.makeAsyncIterator()
                    _ = try #require(await reading.next())
                    let previousEvents = fixture.control.events
                    let invalid = LanguageModelExecutorGenerationRequest(
                        id: UUID(),
                        transcript: Transcript(entries: [.prompt(.init(segments: [.text(.init(content: "b"))]))]),
                        enabledTools: [], generationOptions: .init(), contextOptions: .init(),
                        metadata: ["unsupported": true])
                    await #expect(throws: LanguageModelError.self) {
                        try await fixture.runtime.respond(to: invalid, streamingInto: .init())
                    }
                    #expect(fixture.control.state.withLock { $0.releaseRead != nil })
                    #expect(fixture.control.events == previousEvents)
                    fixture.control.resumeRead()
                    try await group.waitForAll()
                } catch {
                    group.cancelAll()
                    fixture.control.resumeRead()
                    throw error
                }
            }
        } onCancel: {
            fixture.control.resumeRead()
        }
    }
}
