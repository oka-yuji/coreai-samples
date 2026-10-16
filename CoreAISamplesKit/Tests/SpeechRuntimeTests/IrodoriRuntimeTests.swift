import Foundation
import SpeechCore
import Testing

@testable import SpeechRuntime

@Suite(.timeLimit(.minutes(1)))
struct IrodoriRuntimeTests {
    @Test("Opening a model performs no inference and invalid text never loads a pipeline")
    func inputBeforeInference() async throws {
        let driver = TestDriver()
        let runtime = IrodoriRuntime(dependencies: testDependencies(driver))
        await #expect(throws: SpeechError.noModel) { try await runtime.generate(text: "13", seed: 1) }
        let model = try await runtime.open(URL(filePath: "/unused-model"))
        #expect(model.name == "Irodori v4.1 Small MF (FP32)")
        #expect(await driver.factoryCount == 0)
        await #expect(throws: SpeechError.emptyText) { try await runtime.generate(text: "empty", seed: 1) }
        await #expect(throws: SpeechError.textTooLong(256)) { try await runtime.generate(text: "long", seed: 1) }
        #expect(await driver.factoryCount == 0)
    }

    @Test("Repeated generation reuses models and shape changes release the old context first")
    func shapeLifecycle() async throws {
        let driver = TestDriver()
        let runtime = IrodoriRuntime(dependencies: testDependencies(driver))
        _ = try await runtime.open(URL(filePath: "/unused-model"))
        let first = try await runtime.generate(text: "13", seed: 1708)
        #expect(first.audioSeconds == 0.52)
        let expectedBytes: Int = 58 + 13 * 1920 * 4
        #expect(first.wav.count == expectedBytes)
        _ = try await runtime.generate(text: "13", seed: 1709)
        #expect(await driver.factoryCount == 1)
        _ = try await runtime.generate(text: "26", seed: 1710)
        #expect(await driver.factoryCount == 2)
        let events = driver.log.snapshot()
        let second = try #require(events.firstIndex(of: "factory-2"))
        #expect(Array(events[..<second].suffix(2)) == ["context-ended", "pipeline-ended"])
        #expect(await driver.seeds == [1708, 1709, 1710])
        await runtime.release()
        #expect(driver.log.snapshot().last == "pipeline-ended")
    }

    @Test("The sixty-first completed request reloads every model")
    func periodicReload() async throws {
        let driver = TestDriver()
        let runtime = IrodoriRuntime(dependencies: testDependencies(driver))
        _ = try await runtime.open(URL(filePath: "/unused-model"))
        for seed in 0..<61 { _ = try await runtime.generate(text: "13", seed: UInt64(seed)) }
        #expect(await driver.factoryCount == 2)
        await runtime.release()
    }

    @Test("Cancellation keeps release waiting for an uncooperative operation to finish")
    func stopWaitsForOperation() async throws {
        let driver = TestDriver()
        await driver.configure(block: true)
        let releaseStarted = TestSignal()
        let runtime = IrodoriRuntime(dependencies: testDependencies(driver, releasing: releaseStarted))
        _ = try await runtime.open(URL(filePath: "/unused-model"))
        let gate = driver.gate
        try await withThrowingTaskGroup(of: String.self) { group in
            defer { gate.open() }
            group.addTask {
                do {
                    _ = try await runtime.generate(text: "13", seed: 1)
                    return "completed"
                } catch is CancellationError { return "cancelled" }
            }
            try await driver.started.wait()
            await #expect(throws: SpeechError.busy) { try await runtime.generate(text: "13", seed: 2) }
            await #expect(throws: SpeechError.busy) { try await runtime.open(URL(filePath: "/other")) }
            group.addTask {
                await runtime.release()
                return "released"
            }
            try await releaseStarted.wait()
            #expect(!driver.log.snapshot().contains("pipeline-ended"))
            group.cancelAll()
            gate.open()
            var outcomes = Set<String>()
            for try await outcome in group { outcomes.insert(outcome) }
            #expect(outcomes == ["cancelled", "released"])
        }
        await driver.configure()
        _ = try await runtime.generate(text: "13", seed: 3)
        #expect(await driver.factoryCount == 2)
        await runtime.release()
    }

    @Test("A failed generation drops the models and a later request can recover")
    func recovery() async throws {
        let driver = TestDriver()
        await driver.configure(fail: true)
        let runtime = IrodoriRuntime(dependencies: testDependencies(driver))
        _ = try await runtime.open(URL(filePath: "/unused-model"))
        await #expect(throws: SpeechError.invalidSamples) { try await runtime.generate(text: "13", seed: 1) }
        #expect(driver.log.snapshot().last == "pipeline-ended")
        await driver.configure()
        _ = try await runtime.generate(text: "13", seed: 2)
        #expect(await driver.factoryCount == 2)
        await runtime.release()
    }
}
