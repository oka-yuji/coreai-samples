import Foundation
import SpeechCore
import Testing
import VoxCPMCore

@testable import SpeechAssets
@testable import SpeechRuntime

@Suite(.serialized, .timeLimit(.minutes(1)))
struct VoxCPMRuntimeTests {
    static let directory = URL(filePath: "/unused-model")

    static func word(_ data: Data, _ offset: Int) -> UInt32 {
        data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
    }

    static func half(_ data: Data, _ offset: Int) -> UInt16 {
        data.withUnsafeBytes { UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt16.self)) }
    }

    static func reference() throws -> SpeechReferenceAudio {
        try SpeechReferenceAudio(
            samples: [Float](repeating: 0.1, count: SpeechReferenceAudio.minimumSamples), source: .recording)
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

    @Test("VRT-01 A request during an operation is rejected as busy and is not run later")
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

    @Test("VRT-01 While the pipeline loads, a generation and an open are rejected as busy and are not run later")
    func busyDuringLoad() async throws {
        let driver = FakeVoxCPMDriver()
        await driver.configure(blockLoad: true)
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies())
        _ = try await runtime.open(Self.directory)
        let loadGate = driver.loadGate
        try await withThrowingTaskGroup(of: Void.self) { group in
            defer { loadGate.open() }
            group.addTask { _ = try await runtime.generate(text: "ab", seed: 1) }
            try await driver.loadStarted.wait()
            await #expect(throws: SpeechError.busy) { try await runtime.generate(text: "cd", seed: 2) }
            await #expect(throws: SpeechError.busy) { try await runtime.open(URL(filePath: "/other")) }
            #expect(await driver.factoryCount == 0)
            loadGate.open()
            try await group.waitForAll()
        }
        #expect(await driver.factoryCount == 1)
        #expect(await driver.seeds == [1])
        #expect(driver.log.snapshot().filter { $0 == "open" }.count == 1)
        await runtime.release()
    }

    @Test(
        "VRT-02 Each release waits for the operation, cancelled or not, and nothing is written back after it",
        arguments: [true, false])
    func releaseWaitsForOperation(cancelled: Bool) async throws {
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
            if cancelled { group.cancelAll() }
            gate.open()
            var outcomes = Set<String>()
            for try await outcome in group { outcomes.insert(outcome) }
            let operation = cancelled ? "cancelled" : "completed"
            #expect(outcomes == [operation, "first released after the end", "second released after the end"])
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

    @Test("VRT-02 A release during the pipeline load returns after the operation ends and leaves no pipeline behind")
    func releaseDuringLoad() async throws {
        let driver = FakeVoxCPMDriver()
        await driver.configure(blockLoad: true)
        let (requested, request) = AsyncStream<Void>.makeStream()
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies(releasing: { request.yield() }))
        _ = try await runtime.open(Self.directory)
        let loadGate = driver.loadGate
        let log = driver.log
        try await withThrowingTaskGroup(of: String.self) { group in
            defer { loadGate.open() }
            group.addTask {
                _ = try await runtime.generate(text: "ab", seed: 1)
                return "completed"
            }
            try await driver.loadStarted.wait()
            group.addTask {
                await runtime.release()
                return log.snapshot().last == "pipeline-ended" ? "released after the end" : "released early"
            }
            var requests = requested.makeAsyncIterator()
            _ = await requests.next()
            #expect(await driver.factoryCount == 0)
            loadGate.open()
            var outcomes = Set<String>()
            for try await outcome in group { outcomes.insert(outcome) }
            #expect(outcomes == ["completed", "released after the end"])
        }
        #expect(await driver.factoryCount == 1)
        _ = try await runtime.generate(text: "cd", seed: 2)
        #expect(await driver.factoryCount == 2)
        #expect(log.snapshot().filter { $0 == "open" }.count == 1)
        await runtime.release()
    }

    @Test("VRT-03 A failed load or generation drops the pipeline and the next request reloads it without reopening")
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

    @Test("VRT-03 A cancelled generation drops the pipeline and the next request reloads it")
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

    @Test("VRT-04 Text over the prefill limit, a generation at the patch limit and a reference voice return no audio")
    func rejectedRequests() async throws {
        let driver = FakeVoxCPMDriver()
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies())
        _ = try await runtime.open(Self.directory)
        let limit = try VoxCPMPipelineTests.contract().prefillLength - 1
        await #expect(throws: SpeechError.textExceedsLimit(count: limit + 1, limit: limit)) {
            try await runtime.generate(text: String(repeating: "a", count: limit + 1), seed: 1)
        }
        let reference = try Self.reference()
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

    @Test("VRT-04 Requests rejected for their input keep the loaded pipeline for the next generation")
    func rejectionKeepsPipeline() async throws {
        let driver = FakeVoxCPMDriver()
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies())
        _ = try await runtime.open(Self.directory)
        _ = try await runtime.generate(text: "ab", seed: 1)
        let limit = try VoxCPMPipelineTests.contract().prefillLength - 1
        await #expect(throws: SpeechError.textExceedsLimit(count: limit + 1, limit: limit)) {
            try await runtime.generate(text: String(repeating: "a", count: limit + 1), seed: 2)
        }
        await #expect(throws: SpeechError.emptyText) { try await runtime.generate(text: " \n\u{3000}", seed: 3) }
        let reference = try Self.reference()
        await #expect(throws: SpeechError.referenceNotSupported) {
            try await runtime.generate(text: "ab", seed: 4, reference: reference)
        }
        _ = try await runtime.generate(text: "cd", seed: 5)
        #expect(await driver.factoryCount == 1)
        #expect(await driver.seeds == [1, 5])
        #expect(!driver.log.snapshot().contains("pipeline-ended"))
        await runtime.release()
    }

    @Test("VRT-05 The WAV is 48 kHz mono float with the samples of every patch up to the patch limit of the text")
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

extension VoxCPMRuntimeTests {
    static func voice(_ count: Int, _ value: Float = 0.25) throws -> SpeechReferenceAudio {
        try SpeechReferenceAudio(
            samples: [Float](repeating: value, count: count), source: .file("voice.wav"), format: .voxcpm2)
    }

    static func speech(
        patches: Int, reported: Int? = nil, stopped: Int? = nil, extra: Int = 0, rate: Int = 48_000,
        last: Float = 0.25
    ) -> VoxCPMSpeech {
        let count = reported ?? patches
        var samples = [Float](repeating: 0.25, count: patches * 7_680 + extra)
        if !samples.isEmpty { samples[samples.count - 1] = last }
        return VoxCPMSpeech(samples: samples, sampleRate: rate, patchCount: count, stoppedPatch: stopped ?? count - 1)
    }

    static func outcome(of runtime: VoxCPMRuntime, voice: SpeechReferenceAudio) async throws -> String {
        do {
            _ = try await runtime.generate(text: "ab", seed: 1, reference: voice)
            return "completed"
        } catch is CancellationError { return "cancelled" }
    }

    static func expectBusy(_ runtime: VoxCPMRuntime, voice: SpeechReferenceAudio) async {
        let overlap = await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await runtime.generate(text: "", seed: 2, reference: voice)
        }.result
        #expect(throws: SpeechError.busy) { try overlap.get() }
        await #expect(throws: SpeechError.busy) { try await runtime.open(URL(filePath: "/other")) }
    }

    @Test("VRW-01 Only the reference bundle reports a voice and each request reaches the pipeline requirement")
    func referenceForwarding() async throws {
        let driver = FakeVoxCPMDriver()
        let service: any SpeechGenerating = VoxCPMRuntime(
            dependencies: driver.dependencies(profile: .referenceFP32))
        let model = try await service.open(Self.directory)
        #expect(
            model
                == SpeechModel(
                    name: "VoxCPM2 (FP32, reference audio)", directory: Self.directory, supportsReferenceAudio: true))
        #expect(await driver.factoryCount == 0)
        let first = try Self.voice(80_000)
        let second = try Self.voice(160_000, 0.5)
        let voices: [SpeechReferenceAudio?] = [first, nil, second, first]
        for (index, voice) in voices.enumerated() {
            _ = try await service.generate(text: "ab", seed: UInt64(index + 7), reference: voice)
        }
        _ = try await service.generate(text: "ab", seed: 11)
        let body = "ab".unicodeScalars.map { FakeVoxCPMTextTokenizer.offset + Int32($0.value) }
        #expect(await driver.requests == [[Int32]](repeating: body + [VoxCPMPrompt.audioStartTokenID], count: 5))
        #expect(await driver.seeds == [7, 8, 9, 10, 11])
        #expect(await driver.references == voices + [nil])
        #expect(await driver.factoryCount == 1)
        await service.release()
    }

    @Test("VRW-01 The reference pipeline reached through the pipeline requirement encodes the voice before the prompt")
    func pipelineWitness() async throws {
        let log = FakeVoxCPMLog()
        let pipeline: any VoxCPMSpeechPipeline = try VoxCPMReferenceConditioningTests.pipeline(log)
        let voice = try VoxCPMReferenceConditioningTests.voice(seconds: 5)
        let speech = try await pipeline.generate(ids: [8, 101], seed: 1708, reference: voice)
        let pcm = FakeVoxCPMComponents.floats(log.calls.first?.inputs["pcm"])
        #expect(log.calls.first?.function == .referenceEncode)
        #expect(pcm.count == 161_280 && pcm.prefix(voice.samples.count).elementsEqual(voice.samples))
        #expect(speech.patchCount == 4 && speech.stoppedPatch == 3)
        let plain = FakeVoxCPMLog()
        let text: any VoxCPMSpeechPipeline = try VoxCPMReferenceConditioningTests.pipeline(plain)
        _ = try await text.generate(ids: [8, 101], seed: 1708, reference: nil)
        #expect(!plain.calls.isEmpty && !plain.calls.contains { $0.function == .referenceEncode })
    }

    @Test("VRW-02 Reference requests are refused for their input before any load and keep the warm pipeline")
    func referencePreflight() async throws {
        let driver = FakeVoxCPMDriver()
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies(profile: .referenceFP32))
        let short = try Self.voice(80_000)
        let long = try Self.voice(160_000, 0.5)
        await #expect(throws: SpeechError.modelNotOpened("VoxCPM2")) {
            try await runtime.generate(text: "ab", seed: 1, reference: short)
        }
        let cancelled = await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await runtime.generate(text: "", seed: 1, reference: short)
        }.result
        #expect(throws: CancellationError.self) { try cancelled.get() }
        _ = try await runtime.open(Self.directory)
        let limits: [(Int, Int, SpeechReferenceAudio?)] = [(111, 110, short), (80, 79, long), (145, 144, nil)]
        for round in 0..<2 {
            let irodori = try Self.reference()
            await #expect(throws: SpeechReferenceError.incompatibleFormat) {
                try await runtime.generate(text: "ab", seed: 2, reference: irodori)
            }
            await #expect(throws: SpeechError.emptyText) {
                try await runtime.generate(text: " \n\u{3000}", seed: 2, reference: short)
            }
            await #expect(throws: SpeechError.unreadableText) {
                try await runtime.generate(text: "a\u{E9}b", seed: 2, reference: long)
            }
            for (count, limit, voice) in limits {
                await #expect(throws: SpeechError.textExceedsLimit(count: count, limit: limit)) {
                    try await runtime.generate(text: String(repeating: "a", count: count), seed: 3, reference: voice)
                }
            }
            #expect(await driver.factoryCount == round)
            #expect(await driver.requests.count == round * 3)
            for (count, _, voice) in limits {
                _ = try await runtime.generate(
                    text: String(repeating: "a", count: count - 1), seed: 4, reference: voice)
            }
        }
        #expect(await driver.factoryCount == 1)
        #expect(await driver.references == [short, long, nil, short, long, nil])
        #expect(!driver.log.snapshot().contains("pipeline-ended"))
        await runtime.release()
    }

    @Test("VRW-02 Token IDs outside the model vocabulary are refused as unreadable text before any load")
    func outsideVocabulary() async throws {
        let driver = FakeVoxCPMDriver()
        var dependencies = driver.dependencies(profile: .referenceFP32)
        dependencies.tokenizer = { _ in BoundaryTokenizer() }
        let runtime = VoxCPMRuntime(dependencies: dependencies)
        _ = try await runtime.open(Self.directory)
        let voice = try Self.voice(80_000)
        await #expect(throws: SpeechError.unreadableText) { try await runtime.generate(text: "z+", seed: 1) }
        await #expect(throws: SpeechError.unreadableText) {
            try await runtime.generate(text: "-z", seed: 1, reference: voice)
        }
        #expect(await driver.factoryCount == 0)
        _ = try await runtime.generate(text: "zz", seed: 2, reference: voice)
        #expect(await driver.requests == [[73_447, 73_447, VoxCPMPrompt.audioStartTokenID]])
        await runtime.release()
    }

    @Test("VRW-02 A legacy bundle refuses a voice only after the open and busy checks and keeps text speech")
    func legacyReferencePrecedence() async throws {
        let driver = FakeVoxCPMDriver()
        await driver.configure(block: true)
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies(profile: .fp16))
        let voice = try Self.voice(80_000)
        await #expect(throws: SpeechError.modelNotOpened("VoxCPM2")) {
            try await runtime.generate(text: "ab", seed: 1, reference: voice)
        }
        _ = try await runtime.open(Self.directory)
        let gate = driver.gate
        try await withThrowingTaskGroup(of: Void.self) { group in
            defer { gate.open() }
            group.addTask { _ = try await runtime.generate(text: "ab", seed: 1) }
            try await driver.started.wait()
            await #expect(throws: SpeechError.busy) {
                try await runtime.generate(text: "ab", seed: 2, reference: voice)
            }
            gate.open()
            try await group.waitForAll()
        }
        for reference in try [voice, Self.reference()] {
            await #expect(throws: SpeechError.referenceNotSupported) {
                try await runtime.generate(text: "ab", seed: 3, reference: reference)
            }
        }
        #expect(await driver.seeds == [1])
        #expect(await driver.references == [nil])
        #expect(!driver.log.snapshot().contains("pipeline-ended"))
        await runtime.release()
    }

    @Test(
        "VRW-03 A cancelled reference request in a slow load or generation stays busy, starts nothing and releases first",
        arguments: [(true, false), (true, true), (false, false), (false, true)])
    func cancelledReferenceOperation(duringLoad: Bool, fails: Bool) async throws {
        let driver = FakeVoxCPMDriver()
        let late = SpeechError.invalidAsset("late")
        await driver.configure(
            block: !duringLoad, blockLoad: duringLoad, loadFailure: duringLoad && fails ? late : nil,
            failure: !duringLoad && fails ? late : nil)
        let (requested, request) = AsyncStream<Void>.makeStream()
        let runtime = VoxCPMRuntime(
            dependencies: driver.dependencies(profile: .referenceFP32, releasing: { request.yield() }))
        _ = try await runtime.open(Self.directory)
        let voice = try Self.voice(80_000)
        let created = !(duringLoad && fails) ? 1 : 0
        let (log, gate, loadGate) = (driver.log, driver.gate, driver.loadGate)
        try await withThrowingTaskGroup(of: String.self) { group in
            defer {
                loadGate.open()
                gate.open()
            }
            group.addTask { try await Self.outcome(of: runtime, voice: voice) }
            if duringLoad { try await driver.loadStarted.wait() } else { try await driver.started.wait() }
            await Self.expectBusy(runtime, voice: voice)
            var requests = requested.makeAsyncIterator()
            for name in ["first", "second"] {
                group.addTask {
                    await runtime.release()
                    return "\(name) \(log.snapshot().filter { $0 == "pipeline-ended" }.count)"
                }
                _ = await requests.next()
            }
            #expect(!log.snapshot().contains("pipeline-ended"))
            group.cancelAll()
            loadGate.open()
            gate.open()
            var outcomes = Set<String>()
            for try await outcome in group { outcomes.insert(outcome) }
            #expect(outcomes == ["cancelled", "first \(created)", "second \(created)"])
        }
        #expect(await driver.requests.count == (duringLoad ? 0 : 1))
        #expect(log.snapshot().filter { $0 == "tokenizer" }.count == 1)
        await driver.configure()
        _ = try await runtime.generate(text: "cd", seed: 3, reference: voice)
        _ = try await runtime.generate(text: "ef", seed: 4)
        #expect(await driver.factoryCount == created + 1)
        #expect(await driver.seeds == (duringLoad ? [3, 4] : [1, 3, 4]))
        let counts = ["pipeline-ended", "tokenizer", "open"].map { name in log.snapshot().filter { $0 == name }.count }
        #expect(counts == [created, 2, 1])
        await runtime.release()
    }

    @Test("VRW-04 Failed reference generations and malformed outputs return no audio, drop the pipeline and recover")
    func referenceFailures() async throws {
        let driver = FakeVoxCPMDriver()
        let runtime = VoxCPMRuntime(dependencies: driver.dependencies(profile: .referenceFP32))
        _ = try await runtime.open(Self.directory)
        let voice = try Self.voice(80_000)
        let contract = try VoxCPMPipelineTests.contract(.referenceFP32)
        let prompt = try VoxCPMReferencePrompt(
            textAndStartTokenIDs: [1, 2, 101], referencePatchCount: voice.conditioningPositionCount,
            prefillLength: contract.prefillLength)
        let budget = try VoxCPMStopRule.patchLimit(
            prompt: prompt, capacity: contract.capacity, maximumPatches: contract.manifest.generation.maxLen)
        let cases: [((any Error)?, VoxCPMSpeech?, SpeechError)] = [
            (SpeechError.invalidAsset("component"), nil, .invalidAsset("component")),
            (VoxCPMGenerationError.reachedPatchLimit(budget), nil, .speechNotFinished),
            (VoxCPMGenerationError.invalidOutput("refenc.main.mu"), nil, .invalidSamples),
            (nil, Self.speech(patches: 4, reported: .max), .invalidSamples),
            (nil, Self.speech(patches: 0), .invalidSamples),
            (nil, Self.speech(patches: budget + 1), .invalidSamples),
            (nil, Self.speech(patches: 4, stopped: 8), .invalidSamples),
            (nil, Self.speech(patches: 4, extra: 1), .invalidSamples),
            (nil, Self.speech(patches: 4, rate: 16_000), .invalidSamples),
            (nil, Self.speech(patches: 4, last: .nan), .invalidSamples),
        ]
        for (index, (failure, output, expected)) in cases.enumerated() {
            await driver.configure(failure: failure, output: output)
            await #expect(throws: expected, "\(index)") {
                try await runtime.generate(text: "ab", seed: UInt64(index), reference: voice)
            }
            #expect(driver.log.snapshot().filter { $0 == "pipeline-ended" }.count == index + 1, "\(index)")
            await driver.configure(output: Self.speech(patches: budget))
            let audio = try await runtime.generate(text: "ab", seed: 100, reference: voice)
            #expect(audio.wav.count == 58 + budget * contract.samplesPerPatch * 4, "\(index)")
        }
        #expect(await driver.factoryCount == cases.count + 1)
        #expect(driver.log.snapshot().filter { $0 == "open" || $0 == "tokenizer" } == ["open", "tokenizer"])
        await runtime.release()
    }
}

private struct BoundaryTokenizer: VoxCPMTextTokenizer {
    let vocabulary: [Int32: String] = [-1: "-", 73_447: "z", 73_448: "+"]

    func tokenIDs(for text: String) throws -> [Int32] {
        try text.map { character in
            guard let id = vocabulary.first(where: { $0.value == String(character) })?.key else {
                throw VoxCPMError.unknownToken
            }
            return id
        }
    }
}
