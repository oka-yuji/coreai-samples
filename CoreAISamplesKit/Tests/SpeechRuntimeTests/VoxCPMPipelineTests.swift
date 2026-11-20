import Foundation
import SpeechCore
import Testing
import VoxCPMCore

@testable import SpeechAssets
@testable import SpeechRuntime

struct VoxCPMPipelineTests {
    static let generationFunctions: [VoxCPMFunction] = [
        .embed, .prefill, .step, .residualPrefill, .residualStep, .heads, .fuse, .prefillFuse, .encode, .estimate,
        .decode,
    ]

    static func contract(_ profile: VoxCPMBundleProfile = .fp32) throws -> VoxCPMContract {
        try VoxCPMContract(manifest: VoxCPMBundleInventory.manifest(in: VoxCPMBundleFixture.manifest(profile)))
    }

    static func pipeline(_ log: FakeVoxCPMLog, _ behaviors: [FakeVoxCPMBehavior] = [FakeVoxCPMBehavior()]) throws
        -> VoxCPMPipeline<FakeVoxCPMComponents>
    {
        let contract = try contract()
        return VoxCPMPipeline(
            components: FakeVoxCPMComponents(log: log, contract: contract, behaviors: behaviors), contract: contract)
    }

    static func sequence(patches: Int) -> [VoxCPMFunction] {
        var result: [VoxCPMFunction] = [.embed, .prefill, .prefillFuse, .residualPrefill]
        for index in 0..<patches {
            result += [.heads] + Array(repeating: .estimate, count: VoxCPMSolver.estimatorCallsPerPatch)
            result += [.encode, .decode]
            if index < patches - 1 { result += [.step, .fuse, .residualStep] }
        }
        return result
    }

    static func outcome(_ result: Result<VoxCPMSpeech, any Error>) -> String {
        switch result {
        case .success(let speech): "completed \(speech.patchCount)"
        case .failure(let error as VoxCPMGenerationError): "\(error)"
        case .failure(is CancellationError): "cancelled"
        case .failure(let error): "other \(error)"
        }
    }

    private static let floats = FakeVoxCPMComponents.floats
    private static let integers = FakeVoxCPMComponents.integers

    @Test("VPL-01 With fake parts every call follows the contract order and shape and receives the earlier outputs")
    func callOrder() async throws {
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(log, [FakeVoxCPMBehavior(stopPatch: 4)])
        let ids: [Int32] = [11, 12, 13, 101]
        let speech = try await pipeline.generate(ids: ids, seed: 7)
        let calls = log.calls
        try #require(calls.map(\.function) == Self.sequence(patches: 5))
        for call in calls {
            let inputs = pipeline.contract.manifest.assets[call.function.asset]?.functions[call.function.name]?.inputs
            let expected = Dictionary(uniqueKeysWithValues: (inputs ?? []).map { ($0.name, $0.shape.reduce(1, *)) })
            #expect(call.inputs.mapValues(\.count) == expected, "\(call.function)")
        }
        func input(_ call: FakeVoxCPMLog.Call, _ name: String) -> [Float] { Self.floats(call.inputs[name]) }
        func output(_ call: FakeVoxCPMLog.Call, _ name: String) -> [Float] { call.outputs[name] ?? [] }
        func row(_ values: [Float], _ index: Int) -> [Float] {
            let width = VoxCPMSolver.hiddenSize
            return Array(values[(index * width)..<((index + 1) * width)])
        }
        let padded = Self.integers(calls[0].inputs["ids"])
        #expect(Array(padded.prefix(4)) == ids && padded.dropFirst(4).allSatisfy { $0 == 0 })
        #expect(input(calls[1], "embeds") == output(calls[0], "embeds"))
        #expect(input(calls[2], "hidden") == output(calls[1], "hidden"))
        #expect(input(calls[3], "embeds") == output(calls[2], "residual_inputs"))
        #expect(Self.integers(calls[1].inputs["length"]) == [4] && Self.integers(calls[3].inputs["length"]) == [4])
        var lmRow = row(output(calls[1], "hidden"), 3)
        var residualRow = row(output(calls[3], "hidden"), 3)
        var condition = [Float](repeating: 0, count: VoxCPMSolver.patchSize)
        var noise = VoxCPMNoise(seed: 7)
        var samples: [Float] = []
        var next = 4
        for patch in 0..<5 {
            let heads = calls[next]
            #expect(input(heads, "lm_hidden") == lmRow && input(heads, "residual_hidden") == residualRow, "\(patch)")
            var solver = try VoxCPMSolver(
                noise: noise.nextPatch(), ditHidden: output(heads, "dit_hidden"), condition: condition)
            next += 1
            while let expected = solver.estimatorInput {
                let estimate = calls[next]
                #expect(input(estimate, "x") == expected.x && input(estimate, "mu") == expected.mu, "\(patch)")
                #expect(input(estimate, "t") == expected.t && input(estimate, "cond") == expected.cond, "\(patch)")
                #expect(input(estimate, "dt") == expected.dt, "\(patch)")
                try solver.advance(velocity: output(estimate, "velocity"))
                next += 1
            }
            let encode = calls[next]
            let decode = calls[next + 1]
            #expect(input(encode, "patch") == solver.latent && input(decode, "latent") == solver.latent, "\(patch)")
            samples += output(decode, "audio")
            next += 2
            guard patch < 4 else { break }
            let step = calls[next]
            let fuse = calls[next + 1]
            let residualStep = calls[next + 2]
            let position = [Int32(ids.count + patch)]
            #expect(input(step, "embed") == output(encode, "curr_embed"), "\(patch)")
            #expect(input(fuse, "lm_raw") == output(step, "hidden"), "\(patch)")
            #expect(input(fuse, "curr_embed") == output(encode, "curr_embed"), "\(patch)")
            #expect(input(residualStep, "embed") == output(fuse, "residual_input"), "\(patch)")
            #expect(Self.integers(step.inputs["position"]) == position, "\(patch)")
            #expect(Self.integers(residualStep.inputs["position"]) == position, "\(patch)")
            lmRow = output(fuse, "lm_hidden")
            residualRow = output(residualStep, "hidden")
            condition = solver.latent
            next += 3
        }
        #expect(speech.patchCount == 5 && speech.stoppedPatch == 4 && speech.sampleRate == 48_000)
        #expect(speech.samples == samples)
    }

    @Test("VPL-01 The generation ends at the first stop that the minimum length allows", arguments: [1, 3, 6])
    func stopRule(_ stopPatch: Int) async throws {
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(log, [FakeVoxCPMBehavior(stopPatch: stopPatch)])
        let speech = try await pipeline.generate(ids: [11, 12, 101], seed: 3)
        let expected = max(stopPatch, VoxCPMStopRule.minimumLength + 1)
        #expect(speech.stoppedPatch == expected && speech.patchCount == expected + 1)
        #expect(log.calls.map(\.function) == Self.sequence(patches: expected + 1))
    }

    @Test(
        "VPL-02 Each generation allocates new zeroed states and releases them after a completion, failure or cancellation"
    )
    func freshStates() async throws {
        let log = FakeVoxCPMLog()
        let behaviors = [
            FakeVoxCPMBehavior(stopPatch: 3),
            FakeVoxCPMBehavior(stopPatch: 3, nonFinite: .decode, nonFiniteOccurrence: 1),
            FakeVoxCPMBehavior(stopPatch: 3, cancelledCall: 31),
            FakeVoxCPMBehavior(stopPatch: 3),
        ]
        let pipeline = try Self.pipeline(log, behaviors)
        var outcomes: [String] = []
        for _ in behaviors {
            outcomes.append(
                Self.outcome(await Task { try await pipeline.generate(ids: [11, 12, 101], seed: 1) }.result))
        }
        #expect(outcomes == ["completed 4", "nonFiniteValue(\"vae.main.audio\")", "cancelled", "completed 4"])
        for generation in 1...4 {
            let calls = log.calls(in: generation)
            for function in [VoxCPMFunction.prefill, .residualPrefill, .decode] {
                #expect(calls.first { $0.function == function }?.stateSum == 0, "\(generation) \(function)")
            }
            #expect(calls.first { $0.function == .step }?.stateSum == 1, "\(generation)")
            #expect(calls.first { $0.function == .residualStep }?.stateSum == 1, "\(generation)")
        }
        let lifecycle = log.events.filter { if case .called = $0 { false } else { true } }
        #expect(
            lifecycle == [
                .created(1), .released(1), .created(2), .released(2), .created(3), .released(3), .created(4),
                .released(4),
            ])
    }

    @Test(
        "VPL-03 A cancellation during a part call stops the generation before the next part",
        arguments: [0, 1, 3, 4, 5, 13, 14, 15, 16, 17, 18, 40])
    func cancellationDuringCall(_ call: Int) async throws {
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(log, [FakeVoxCPMBehavior(stopPatch: 3, cancelledCall: call)])
        let result = await Task { try await pipeline.generate(ids: [11, 12, 101], seed: 1) }.result
        #expect(Self.outcome(result) == "cancelled")
        #expect(log.calls.count == call + 1)
        #expect(log.events.last == .released(1))
    }

    @Test("VPL-03 A part that fails after a cancellation reports the cancellation and other part failures pass through")
    func failureAfterCancellation() async throws {
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(
            log, [FakeVoxCPMBehavior(failedCall: 16), FakeVoxCPMBehavior(cancelledCall: 16, failedCall: 16)])
        var outcomes: [String] = []
        for _ in 0..<2 {
            outcomes.append(
                Self.outcome(await Task { try await pipeline.generate(ids: [11, 12, 101], seed: 1) }.result))
        }
        #expect(outcomes == ["invalidOutput(\"fake\")", "cancelled"])
        #expect(log.calls(in: 1).count == 17 && log.calls(in: 2).count == 17)
    }

    @Test("VPL-03 A cancellation before the start or between two parts calls no further part")
    func cancellationBetweenCalls() async throws {
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(log)
        let before = await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await pipeline.generate(ids: [11, 12, 101], seed: 1)
        }.result
        #expect(Self.outcome(before) == "cancelled")
        #expect(log.calls.isEmpty)
        let between = await Task {
            try await pipeline.generate(ids: [11, 12, 101], seed: 1) { record in
                if record.index == 1 { withUnsafeCurrentTask { $0?.cancel() } }
                return nil
            }
        }.result
        #expect(Self.outcome(between) == "cancelled")
        #expect(log.calls.map(\.function) == Array(Self.sequence(patches: 2).prefix(4 + 15 + 10)))
    }

    @Test("VPL-05 A generation that reaches the patch limit without a stop throws and returns no audio")
    func patchLimit() async throws {
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(log, [FakeVoxCPMBehavior(stopPatch: nil)])
        let result = await Task { try await pipeline.generate(ids: [11, 101], seed: 1) }.result
        #expect(Self.outcome(result) == "reachedPatchLimit(16)")
        #expect(log.calls.map(\.function) == Self.sequence(patches: 16))
    }

    @Test(
        "VPL-05 A non-finite part output fails the generation before any later part",
        arguments: VoxCPMPipelineTests.generationFunctions)
    func nonFiniteOutput(_ function: VoxCPMFunction) async throws {
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(log, [FakeVoxCPMBehavior(stopPatch: 3, nonFinite: function)])
        let result = await Task { try await pipeline.generate(ids: [11, 12, 101], seed: 1) }.result
        let outputs = pipeline.contract.manifest.assets[function.asset]?.functions[function.name]?.outputs ?? []
        let name = try #require(outputs.map(\.name).sorted().first)
        #expect(Self.outcome(result) == "nonFiniteValue(\"\(function.rawValue).\(name)\")")
        #expect(log.calls.last?.function == function)
        #expect(log.calls.filter { $0.function == function }.count == 1)
    }

    @Test("VRA-02 Text generation calls only the eleven original functions and refenc.main is the twelfth")
    func twelfthFunction() async throws {
        #expect(VoxCPMFunction.allCases == Self.generationFunctions + [.referenceEncode])
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(log)
        _ = try await pipeline.generate(ids: [11, 12, 101], seed: 1)
        #expect(Set(log.calls.map(\.function)) == Set(Self.generationFunctions))
        #expect(VoxCPMFunction.encode.rawValue == "locenc.main" && VoxCPMFunction.encode.asset == "locenc")
        #expect(VoxCPMFunction.referenceEncode.rawValue == "refenc.main")
        #expect(VoxCPMFunction.referenceEncode.asset == "refenc" && VoxCPMFunction.referenceEncode.name == "main")
    }

    @Test("VPL-05 Token IDs outside the prefill or without the audio start token are rejected before any part call")
    func invalidInput() async throws {
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(log)
        let long = [Int32](repeating: 11, count: 145) + [101]
        let cases: [([Int32], String)] = [
            ([], "invalidTokenIDs"), ([101], "invalidTokenIDs"), ([11, 12], "invalidTokenIDs"),
            ([-1, 101], "invalidTokenIDs"), (long, "textTooLong(145)"),
        ]
        for (ids, expected) in cases {
            let result = await Task { try await pipeline.generate(ids: ids, seed: 1) }.result
            #expect(Self.outcome(result) == expected, "\(ids.count)")
        }
        #expect(log.calls.isEmpty)
        let speech = try await pipeline.generate(ids: Array(long.dropFirst()), seed: 1)
        #expect(speech.patchCount == 4)
    }
}
