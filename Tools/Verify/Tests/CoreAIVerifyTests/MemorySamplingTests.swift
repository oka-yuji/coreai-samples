import ArgumentParser
import Foundation
import Synchronization
import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct MemorySamplingTests {
    private func arguments(runtime: String = "own", turns: [Int] = []) -> [String] {
        ["--model", "/not-loaded", "--runtime", runtime]
            + (1...30).flatMap { ["--prompt", "Turn \($0)"] }
            + turns.flatMap { ["--memory-sample-turn", String($0)] }
    }

    @Test("Memory sampling is absent by default and accepts explicit observation turns")
    func options() throws {
        #expect(try Chat.parse(arguments()).memorySampleTurn.isEmpty)
        #expect(try Chat.parse(arguments(runtime: "official")).memorySampleTurn.isEmpty)
        #expect(try Chat.parse(arguments(turns: [10, 20, 30])).memorySampleTurn == [10, 20, 30])
    }

    @Test(
        "Duplicate and nonexistent observation turns are rejected before loading",
        arguments: [[10, 10], [0], [-1], [31]])
    func invalidTurns(_ turns: [Int]) {
        #expect(throws: (any Error).self) {
            try Chat.parse(arguments(turns: turns))
        }
    }

    @Test("The official runtime rejects requested memory observations before loading")
    func officialRuntime() {
        #expect(throws: (any Error).self) {
            try Chat.parse(arguments(runtime: "official", turns: [10]))
        }
    }

    @Test(
        "The observation waits until five seconds after its start before capturing memory and reports a separate record"
    )
    func waitsThenCaptures() async throws {
        let events = Mutex<[String]>([])
        let started = ContinuousClock.now
        let sample = try await sampleMemory(
            turn: 20, retaining: 1, started: started,
            waitUntil: { deadline in
                #expect(deadline == started + .seconds(5))
                events.withLock { $0.append("wait") }
            },
            capture: {
                events.withLock { $0.append("memory") }
                return Memory(rssBytes: 100, footprintBytes: 80, graphicsFootprintBytes: 60)
            })
        #expect(events.withLock { $0 } == ["wait", "memory"])
        #expect(sample.turn == 20)
        #expect(sample.waitSeconds == 5)
        #expect(sample.elapsedSeconds >= 0)
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let record = try #require(JSONSerialization.jsonObject(with: encoder.encode(sample)) as? [String: Any])
        #expect(record["record"] as? String == "memory-sample")
        #expect(record["runtime"] as? String == "own")
        #expect(record["turn"] as? Int == 20)
        #expect(record["wait_seconds"] as? Int == 5)
        #expect(record["elapsed_seconds"] as? Double == sample.elapsedSeconds)
        let memory = try #require(record["memory"] as? [String: Int])
        #expect(memory == ["rss_bytes": 100, "footprint_bytes": 80, "graphics_footprint_bytes": 60])
    }

    @Test("Cancellation while waiting does not capture memory")
    func cancelledWait() async {
        let captured = Mutex(false)
        await #expect(throws: CancellationError.self) {
            try await sampleMemory(
                turn: 10, retaining: 1,
                waitUntil: { _ in throw CancellationError() },
                capture: {
                    captured.withLock { $0 = true }
                    return Memory(rssBytes: 0, footprintBytes: 0, graphicsFootprintBytes: 0)
                })
        }
        #expect(!captured.withLock { $0 })
    }

    private enum Failure: Error { case wait }

    @Test("A wait failure propagates without capturing memory")
    func failedWait() async {
        let captured = Mutex(false)
        await #expect(throws: Failure.wait) {
            try await sampleMemory(
                turn: 10, retaining: 1,
                waitUntil: { _ in throw Failure.wait },
                capture: {
                    captured.withLock { $0 = true }
                    return Memory(rssBytes: 0, footprintBytes: 0, graphicsFootprintBytes: 0)
                })
        }
        #expect(!captured.withLock { $0 })
    }
}
