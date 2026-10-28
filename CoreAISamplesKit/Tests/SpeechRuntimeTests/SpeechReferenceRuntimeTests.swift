import Foundation
import SpeechCore
import Testing

@testable import SpeechRuntime

@Suite(.timeLimit(.minutes(1)))
struct SpeechReferenceRuntimeTests {
    @Test("REF-CORE-02 Legacy models reject a reference voice before loading a pipeline")
    func legacyRejection() async throws {
        let driver = TestDriver()
        let runtime = IrodoriRuntime(dependencies: testDependencies(driver))
        let model = try await runtime.open(URL(filePath: "/fixture"))
        #expect(!model.supportsReferenceAudio)
        let reference = try voice(24_000)
        await #expect(throws: SpeechError.referenceNotSupported) {
            try await runtime.generate(text: "13", seed: 1, reference: reference)
        }
        #expect(await driver.factoryCount == 0)
        _ = try await runtime.generate(text: "13", seed: 1)
        #expect(await driver.factoryCount == 1)
        await runtime.release()
    }

    @Test("REF-CORE-02 Reference samples reach preparation and changing their shape recycles the pipeline")
    func forwardingAndShapes() async throws {
        let driver = TestDriver()
        let runtime = IrodoriRuntime(dependencies: testDependencies(driver, referenceVoice: true))
        let model = try await runtime.open(URL(filePath: "/fixture"))
        #expect(model.supportsReferenceAudio)
        let first = try voice(24_000)
        let withinPatch = try voice(25_000)
        let second = try voice(48_000)
        _ = try await runtime.generate(text: "13", seed: 1, reference: first)
        _ = try await runtime.generate(text: "13", seed: 2, reference: first)
        #expect(await driver.factoryCount == 1)
        _ = try await runtime.generate(text: "13", seed: 3, reference: withinPatch)
        #expect(await driver.factoryCount == 2)
        _ = try await runtime.generate(text: "13", seed: 3, reference: second)
        #expect(await driver.factoryCount == 3)
        #expect(await driver.references == [first, first, withinPatch, second])
        _ = try await runtime.generate(text: "13", seed: 4)
        #expect(await driver.factoryCount == 4)
        #expect(await driver.references.last == .some(nil))
        await runtime.release()
    }

    @Test("REF-CORE-03 Cancelling reference generation joins release and permits a fresh voice afterwards")
    func cancellation() async throws {
        let driver = TestDriver()
        await driver.configure(block: true)
        let releaseStarted = TestSignal()
        let runtime = IrodoriRuntime(
            dependencies: testDependencies(driver, releasing: releaseStarted, referenceVoice: true))
        _ = try await runtime.open(URL(filePath: "/fixture"))
        let reference = try voice(24_000)
        try await withThrowingTaskGroup(of: String.self) { group in
            defer { driver.gate.open() }
            group.addTask {
                do {
                    _ = try await runtime.generate(text: "13", seed: 1, reference: reference)
                    return "completed"
                } catch is CancellationError { return "cancelled" }
            }
            try await driver.started.wait()
            group.addTask {
                await runtime.release()
                return "released"
            }
            try await releaseStarted.wait()
            #expect(!driver.log.snapshot().contains("pipeline-ended"))
            group.cancelAll()
            driver.gate.open()
            var results = Set<String>()
            for try await result in group { results.insert(result) }
            #expect(results == ["cancelled", "released"])
        }
        await driver.configure()
        _ = try await runtime.generate(text: "13", seed: 2, reference: try voice(48_000))
        #expect(await driver.factoryCount == 2)
        await runtime.release()
    }

    private func voice(_ count: Int) throws -> SpeechReferenceAudio {
        try SpeechReferenceAudio(samples: [Float](repeating: 0.25, count: count), source: .recording)
    }
}
