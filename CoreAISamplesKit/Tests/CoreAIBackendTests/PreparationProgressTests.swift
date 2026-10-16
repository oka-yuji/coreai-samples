import Foundation
import FoundationModels
import Testing

@testable import CoreAIBackend

@Suite(.timeLimit(.minutes(1)))
struct PreparationProgressTests {
    @Test("Preparation reports token counts while the runtime is still reading the restored prompt")
    func reportsWhileReading() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 0))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        let transcript = Transcript(entries: [
            .prompt(.init(id: "old", segments: [.text(.init(content: "a"))])),
            .response(.init(segments: [.text(.init(content: "b"))])),
        ])
        let session = LanguageModelSession(model: fixture.model, transcript: transcript)
        var updates = fixture.model.preparationUpdates().makeAsyncIterator()
        fixture.control.state.withLock { $0.blockNextRead = true }
        try await withThrowingTaskGroup(of: Void.self) { group in
            defer { fixture.control.resumeRead() }
            group.addTask { _ = try await session.respond(to: "aaa") }
            var reading = fixture.control.reading.stream.makeAsyncIterator()
            _ = await reading.next()
            let started = try #require(await updates.next())
            #expect(
                started.transcriptID == "old" && started.processedTokens == 0 && started.totalTokens == 4,
                "Actual: \(started)")
            fixture.control.resumeRead()
            try await group.waitForAll()
            let ended = try #require(await updates.next())
            #expect(
                ended.requestID == started.requestID && ended.processedTokens == 4 && ended.totalTokens == 4,
                "Actual: \(ended)")
        }
    }

    @Test("Cancelling preparation prevents decoding and leaves the runtime available for another request")
    func cancelsReading() async throws {
        let fixture = try await ReloadTestModel.make(policy: .init(readInterval: 0))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        fixture.control.state.withLock { $0.blockNextRead = true }
        await withTaskGroup(of: Bool.self) { group in
            defer { fixture.control.resumeRead() }
            group.addTask {
                do {
                    try await fixture.runtime.respond(
                        to: directReloadRequest(
                            Transcript(entries: [
                                .prompt(.init(id: "p", segments: [.text(.init(content: "aaa"))]))
                            ])), streamingInto: .init())
                    return false
                } catch is CancellationError { return true } catch { return false }
            }
            var reading = fixture.control.reading.stream.makeAsyncIterator()
            _ = await reading.next()
            group.cancelAll()
            fixture.control.resumeRead()
            #expect(await group.next() == true)
        }
        #expect(await fixture.model.processedTokens().isEmpty)
        #expect(try await fixture.request().content == "b")
    }
}
