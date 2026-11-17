import Foundation
import SpeechCore
import Testing
import VoxCPMCore

@testable import SpeechAssets
@testable import SpeechRuntime

@Suite(.timeLimit(.minutes(1)))
struct VoxCPMRuntimeTests {
    static let directory = URL(filePath: "/unused-model")

    static func word(_ data: Data, _ offset: Int) -> UInt32 {
        data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
    }

    static func half(_ data: Data, _ offset: Int) -> UInt16 {
        data.withUnsafeBytes { UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt16.self)) }
    }

    @Test(
        "Opening validates the bundle once, loads no pipeline and names the configuration from the manifest",
        arguments: [
            (VoxCPMBundleProfile.fp32, "VoxCPM2 (FP32)"), (.fp16, "VoxCPM2 (FP16)"),
            (.fp16Proj32, "VoxCPM2 (FP16, FP32 projection)"),
        ])
    func openWithoutInference(profile: VoxCPMBundleProfile, name: String) async throws {
        let driver = FakeVoxCPMDriver()
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies(profile: profile))
        await #expect(throws: SpeechError.modelNotOpened("VoxCPM2")) { try await runtime.generate(text: "ab", seed: 1) }
        let model = try await runtime.open(Self.directory)
        #expect(model == SpeechModel(name: name, directory: Self.directory, supportsReferenceAudio: false))
        #expect(driver.log.snapshot() == ["open", "tokenizer"])
        #expect(await driver.factoryCount == 0)
        _ = try await runtime.generate(text: "ab", seed: 1)
        _ = try await runtime.generate(text: "cd", seed: 2)
        #expect(await driver.factoryCount == 1)
        #expect(driver.log.snapshot().filter { $0 == "open" || $0 == "tokenizer" } == ["open", "tokenizer"])
        await runtime.release()
        #expect(driver.log.snapshot().last == "pipeline-ended")
        _ = try await runtime.generate(text: "ab", seed: 3)
        #expect(await driver.factoryCount == 2)
        #expect(driver.log.snapshot().filter { $0 == "tokenizer" }.count == 2)
        #expect(driver.log.snapshot().filter { $0 == "open" }.count == 1)
        await runtime.release()
    }

    @Test("Generation tokenizes the normalized text and passes only the prompt built from those IDs with the seed")
    func tokenizedPrompt() async throws {
        let driver = FakeVoxCPMDriver()
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies())
        _ = try await runtime.open(Self.directory)
        let normalized = " (calm voice) a b "
        let audio = try await runtime.generate(text: " (calm\u{3000}voice)\n\na\t\u{A0}b\u{2028}", seed: 1708)
        #expect(audio.normalizedText == normalized)
        #expect(driver.log.snapshot().contains("tokenize:" + normalized))
        let expected = normalized.unicodeScalars.map { FakeVoxCPMTextTokenizer.offset + Int32($0.value) }
        #expect(await driver.requests == [expected + [VoxCPMPrompt.audioStartTokenID]])
        #expect(await driver.seeds == [1708])
        await #expect(throws: SpeechError.emptyText) { try await runtime.generate(text: " \n\u{3000}\t", seed: 1) }
        await #expect(throws: SpeechError.emptyText) { try await runtime.generate(text: "", seed: 1) }
        #expect(await driver.requests.count == 1)
        await runtime.release()
    }

    @Test("A request during an operation is rejected as busy and is not run later")
    func busyRejection() async throws {
        let driver = FakeVoxCPMDriver()
        await driver.configure(block: true)
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies())
        _ = try await runtime.open(Self.directory)
        let gate = driver.gate
        try await withThrowingTaskGroup(of: Void.self) { group in
            defer { gate.open() }
            group.addTask { _ = try await runtime.generate(text: "ab", seed: 1) }
            try await driver.started.wait()
            await #expect(throws: SpeechError.busy) { try await runtime.generate(text: "cd", seed: 2) }
            await #expect(throws: SpeechError.busy) { try await runtime.open(URL(filePath: "/other")) }
            gate.open()
            try await group.waitForAll()
        }
        #expect(await driver.seeds == [1])
        await driver.configure()
        _ = try await runtime.generate(text: "cd", seed: 3)
        #expect(await driver.seeds == [1, 3])
        #expect(await driver.factoryCount == 1)
        await runtime.release()
    }

    @Test("Every release waits for an operation that ignores cancellation and nothing is written back after it")
    func releaseWaitsForOperation() async throws {
        let driver = FakeVoxCPMDriver()
        await driver.configure(block: true)
        let (requested, request) = AsyncStream<Void>.makeStream()
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies(releasing: { request.yield() }))
        _ = try await runtime.open(Self.directory)
        let gate = driver.gate
        let log = driver.log
        try await withThrowingTaskGroup(of: String.self) { group in
            defer { gate.open() }
            group.addTask {
                do {
                    _ = try await runtime.generate(text: "ab", seed: 1)
                    return "completed"
                } catch is CancellationError { return "cancelled" }
            }
            try await driver.started.wait()
            var requests = requested.makeAsyncIterator()
            for name in ["first", "second"] {
                group.addTask {
                    await runtime.release()
                    return log.snapshot().last == "pipeline-ended" ? "\(name) released after the end" : "\(name) early"
                }
                _ = await requests.next()
            }
            #expect(!log.snapshot().contains("pipeline-ended"))
            group.cancelAll()
            gate.open()
            var outcomes = Set<String>()
            for try await outcome in group { outcomes.insert(outcome) }
            #expect(outcomes == ["cancelled", "first released after the end", "second released after the end"])
        }
        #expect(log.snapshot().filter { $0 == "pipeline-ended" }.count == 1)
        #expect(log.snapshot().last == "pipeline-ended")
        await driver.configure()
        _ = try await runtime.generate(text: "cd", seed: 2)
        #expect(await driver.factoryCount == 2)
        #expect(log.snapshot().filter { $0 == "open" }.count == 1)
        #expect(log.snapshot().filter { $0 == "tokenizer" }.count == 2)
        await runtime.release()
    }

    @Test("A failed load or generation drops the pipeline and the next request reloads it without reopening")
    func recoveryAfterFailure() async throws {
        let driver = FakeVoxCPMDriver()
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies())
        _ = try await runtime.open(Self.directory)
        await driver.configure(loadFailure: SpeechError.invalidAsset("load"))
        await #expect(throws: SpeechError.invalidAsset("load")) { try await runtime.generate(text: "ab", seed: 1) }
        #expect(await driver.factoryCount == 0)
        await driver.configure(failure: VoxCPMGenerationError.nonFiniteValue("vae.decode.audio"))
        await #expect(throws: SpeechError.invalidSamples) { try await runtime.generate(text: "ab", seed: 2) }
        #expect(driver.log.snapshot().last == "pipeline-ended")
        await driver.configure()
        _ = try await runtime.generate(text: "ab", seed: 3)
        #expect(await driver.factoryCount == 2)
        #expect(driver.log.snapshot().filter { $0 == "open" }.count == 1)
        await runtime.release()
    }

    @Test("A cancelled generation drops the pipeline and the next request reloads it")
    func recoveryAfterCancellation() async throws {
        let driver = FakeVoxCPMDriver()
        await driver.configure(block: true)
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies())
        _ = try await runtime.open(Self.directory)
        let gate = driver.gate
        try await withThrowingTaskGroup(of: String.self) { group in
            defer { gate.open() }
            group.addTask {
                do {
                    _ = try await runtime.generate(text: "ab", seed: 1)
                    return "completed"
                } catch is CancellationError { return "cancelled" }
            }
            try await driver.started.wait()
            group.cancelAll()
            gate.open()
            #expect(try await group.next() == "cancelled")
        }
        #expect(driver.log.snapshot().last == "pipeline-ended")
        await driver.configure()
        _ = try await runtime.generate(text: "ab", seed: 2)
        #expect(await driver.factoryCount == 2)
        await runtime.release()
    }

    @Test("Text over the prefill limit, a generation at the patch limit and a reference voice return no audio")
    func rejectedRequests() async throws {
        let driver = FakeVoxCPMDriver()
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies())
        _ = try await runtime.open(Self.directory)
        let limit = try VoxCPMPipelineTests.contract().prefillLength - 1
        await #expect(throws: SpeechError.textExceedsLimit(count: limit + 1, limit: limit)) {
            try await runtime.generate(text: String(repeating: "a", count: limit + 1), seed: 1)
        }
        let reference = try SpeechReferenceAudio(
            samples: [Float](repeating: 0.1, count: SpeechReferenceAudio.minimumSamples), source: .recording)
        await #expect(throws: SpeechError.referenceNotSupported) {
            try await runtime.generate(text: "ab", seed: 1, reference: reference)
        }
        #expect(await driver.requests.isEmpty)
        #expect(await driver.factoryCount == 0)
        _ = try await runtime.generate(text: String(repeating: "a", count: limit), seed: 2)
        await driver.configure(failure: VoxCPMGenerationError.reachedPatchLimit(22))
        await #expect(throws: SpeechError.speechNotFinished) { try await runtime.generate(text: "ab", seed: 3) }
        #expect(await driver.requests.count == 2)
        await runtime.release()
    }

    @Test("The WAV is 48 kHz mono float with the samples of every patch up to the patch limit of the text")
    func patchWAV() async throws {
        let driver = FakeVoxCPMDriver()
        await driver.configure(patches: 4)
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies())
        _ = try await runtime.open(Self.directory)
        let contract = try VoxCPMPipelineTests.contract()
        let samples = 4 * contract.samplesPerPatch
        let audio = try await runtime.generate(text: "ab", seed: 1)
        #expect(Self.half(audio.wav, 20) == 3)
        #expect(Self.half(audio.wav, 22) == 1)
        #expect(Self.word(audio.wav, 24) == 48_000)
        #expect(Self.word(audio.wav, 46) == UInt32(samples))
        #expect(Self.word(audio.wav, 54) == UInt32(samples * 4))
        #expect(audio.wav.count == 58 + samples * 4)
        #expect(audio.audioSeconds == Double(samples) / 48_000)
        await driver.configure(patches: 4, extraSamples: 1)
        await #expect(throws: SpeechError.invalidSamples) { try await runtime.generate(text: "ab", seed: 2) }
        let limit = try VoxCPMStopRule.patchLimit(textTokenCount: 2, capacity: contract.capacity)
        await driver.configure(patches: limit)
        let longest = try await runtime.generate(text: "ab", seed: 3)
        #expect(longest.wav.count == 58 + limit * contract.samplesPerPatch * 4)
        await driver.configure(patches: limit + 1)
        await #expect(throws: SpeechError.invalidSamples) { try await runtime.generate(text: "ab", seed: 4) }
        await runtime.release()
    }
}
