import Foundation
import FoundationModels
import Testing

@_spi(Verification) @testable import CoreAIBackend

@Suite(.timeLimit(.minutes(1)))
struct RuntimeGraphicsReloadTests {
    static let graphicsOnly = LocalLanguageModel.ReloadPolicy(
        readInterval: 0, footprintLimitBytes: 0, graphicsGrowthLimitBytes: 100)

    private func turn(
        _ fixture: ReloadTestModel, _ session: LanguageModelSession, graphics: UInt64?, footprint: UInt64 = 0
    ) async throws -> Double? {
        fixture.control.state.withLock {
            $0.graphics = graphics ?? 0
            $0.failGraphics = graphics == nil
            $0.footprint = footprint
        }
        _ = try await session.respond(to: "b", options: .init(samplingMode: .greedy, maximumResponseTokens: 1))
        let response = session.transcript.reversed().compactMap { entry -> Transcript.Response? in
            if case .response(let response) = entry { return response }
            return nil
        }.first
        return try response?.metadata["reloadSeconds"].map { try $0.value(Double.self) }
    }

    private func failRead(_ fixture: ReloadTestModel, _ end: String) async throws {
        guard end == "cancellation" else {
            fixture.control.state.withLock {
                if end == "failure" { $0.failNextRead = true } else { $0.failReadAfterTokens = 1 }
            }
            await #expect(throws: ReloadTestFailure.self) { try await fixture.request("aa") }
            return
        }
        fixture.control.state.withLock { $0.blockNextRead = true }
        try await withTaskCancellationHandler {
            try await withThrowingTaskGroup(of: Bool.self) { group in
                defer { fixture.control.resumeRead() }
                group.addTask {
                    do {
                        try await fixture.runtime.respond(
                            to: directReloadRequest(
                                Transcript(entries: [.prompt(.init(segments: [.text(.init(content: "aa"))]))])),
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
    }

    @Test("When the graphics growth is at or above its limit, a reload is chosen before the next read")
    func growthAtLimitReloadsBeforeTheRead() async throws {
        let fixture = try await ReloadTestModel.make(policy: Self.graphicsOnly)
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        let session = LanguageModelSession(model: fixture.model)
        var reloads: [Double?] = []
        for graphics: UInt64 in [1_000, 1_000, 1_099, 1_100] {
            reloads.append(try await turn(fixture, session, graphics: graphics))
        }
        #expect(reloads.map { $0 != nil } == [false, false, false, true])
        #expect(reloads.last.flatMap { $0 }.map { $0 >= 0 } == true)
        #expect(fixture.control.loads == 2)
        let afterLoad = fixture.control.events.drop { $0 != "load:2" }.filter { $0.hasPrefix("read:") }
        #expect(!afterLoad.isEmpty && afterLoad.allSatisfy { $0 == "read:2" })
    }

    @Test("When the growth is below its limit or cannot be sampled, only the existing two conditions decide")
    func belowLimitOrUnsampledKeepsExistingConditions() async throws {
        let fixture = try await ReloadTestModel.make(
            policy: .init(readInterval: 0, footprintLimitBytes: 1_000, graphicsGrowthLimitBytes: 100))
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        let session = LanguageModelSession(model: fixture.model)
        var reloads: [Double?] = []
        for graphics: UInt64? in [0, nil, 500, 599] {
            reloads.append(try await turn(fixture, session, graphics: graphics))
        }
        #expect(reloads.map { $0 != nil } == [false, false, false, false])
        #expect(fixture.control.loads == 1)
        let footprintReload = try await turn(fixture, session, graphics: nil, footprint: 1_000)
        #expect(footprintReload != nil)
        #expect(fixture.control.loads == 2)
    }

    @Test("The baseline waits for a finished read and is retaken after a reload, so the next turn does not reload")
    func baselineFollowsTheFinishedReadAndEachReload() async throws {
        let fixture = try await ReloadTestModel.make(policy: Self.graphicsOnly)
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        let session = LanguageModelSession(model: fixture.model)
        var reloads: [Bool] = []
        for graphics: UInt64 in [0, 1_000, 1_100, 5_000, 5_099, 5_100] {
            reloads.append(try await turn(fixture, session, graphics: graphics) != nil)
        }
        #expect(reloads == [false, false, true, false, false, true])
        #expect(fixture.control.loads == 3)
    }

    @Test(
        "A zero graphics limit never samples graphics, and a nonzero one samples it even with the footprint limit off",
        arguments: [
            (LocalLanguageModel.ReloadPolicy(readInterval: 0, footprintLimitBytes: 0), 0),
            (LocalLanguageModel.ReloadPolicy(readInterval: 0, footprintLimitBytes: 0, graphicsGrowthLimitBytes: 0), 0),
            (graphicsOnly, 3),
        ])
    func samplesOnlyWithANonzeroLimit(policy: LocalLanguageModel.ReloadPolicy, samples: Int) async throws {
        let fixture = try await ReloadTestModel.make(policy: policy)
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        let session = LanguageModelSession(model: fixture.model)
        for _ in 0..<3 { _ = try await turn(fixture, session, graphics: 1_000) }
        #expect(fixture.control.graphicsSamples == samples)
        #expect(fixture.control.loads == 1)
    }

    @Test(
        "A request whose read fails or is cancelled is not the finished read that the baseline waits for",
        arguments: ["failure", "partial failure", "cancellation"])
    func unfinishedReadTakesNoBaseline(end: String) async throws {
        let fixture = try await ReloadTestModel.make(policy: Self.graphicsOnly)
        defer { try? FileManager.default.removeItem(at: fixture.folder) }
        fixture.control.state.withLock { $0.graphics = 1_000 }
        try await failRead(fixture, end)
        let session = LanguageModelSession(model: fixture.model)
        var reloads: [Bool] = []
        for graphics: UInt64 in [1_000, 1_100, 1_200] {
            reloads.append(try await turn(fixture, session, graphics: graphics) != nil)
        }
        #expect(reloads == [false, false, true])
        #expect(fixture.control.loads == 2)
    }
}
