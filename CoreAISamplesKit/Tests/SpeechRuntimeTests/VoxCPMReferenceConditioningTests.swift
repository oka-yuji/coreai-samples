import Foundation
import SpeechCore
import Testing
import VoxCPMCore

@testable import SpeechAssets
@testable import SpeechRuntime

struct VoxCPMReferenceConditioningTests {
    @Test(
        "VRP-02 Encoder frames reach the local encoder by channel and only the audio rows replace the text embedding",
        arguments: [(5, 32), (10, 63)])
    func referenceRows(seconds: Int, patches: Int) async throws {
        let log = FakeVoxCPMLog()
        let voice = try Self.voice(seconds: seconds)
        let speech = try await Self.pipeline(log).generate(ids: Self.body + [101], seed: 1708, reference: voice)
        let (prefill, _) = try Self.phases(log.calls)
        try #require(prefill.map(\.function) == Self.prefillOrder(patches: patches))
        let pcm = Self.floats(prefill[0].inputs["pcm"])
        #expect(pcm.count == 161_280 && Array(pcm.prefix(voice.samples.count)) == voice.samples)
        #expect(pcm.dropFirst(voice.samples.count).allSatisfy { $0.bitPattern == 0 })
        let mu = prefill[0].outputs["mu"] ?? []
        let encodes = Array(prefill[1...patches])
        for (patch, encode) in encodes.enumerated() {
            let expected = (0..<64).flatMap { channel in (0..<4).map { mu[channel * 252 + patch * 4 + $0] } }
            #expect(Self.floats(encode.inputs["patch"]) == expected, "\(patch)")
        }
        let ids = [103] + [Int32](repeating: 0, count: patches) + [104] + Self.body + [101]
        let embed = prefill[patches + 1]
        #expect(Self.integers(embed.inputs["ids"]) == ids + [Int32](repeating: 0, count: 145 - ids.count))
        let embeds = embed.outputs["embeds"] ?? []
        let text = prefill[patches + 2]
        let input = Self.floats(text.inputs["embeds"])
        try #require(input.count == 145 * 2_048)
        for row in 0..<145 {
            let audio = (1...patches).contains(row)
            let expected = audio ? encodes[row - 1].outputs["curr_embed"] ?? [] : Self.row(embeds, row)
            #expect(Self.row(input, row) == expected && audio == (expected != Self.row(embeds, row)), "\(row)")
        }
        #expect(Self.integers(text.inputs["length"]) == [Int32(ids.count)])
        #expect(speech.patchCount == 4 && speech.stoppedPatch == 3)
    }

    @Test("VRP-02 Each audio row quantizes its raw row with its encoding and only that residual replaces the bulk row")
    func residualRows() async throws {
        let log = FakeVoxCPMLog()
        _ = try await Self.pipeline(log).generate(ids: Self.body + [101], seed: 1708, reference: Self.voice(seconds: 5))
        let (prefill, _) = try Self.phases(log.calls)
        try #require(prefill.map(\.function) == Self.prefillOrder(patches: 32))
        let encodes = Array(prefill[1...32])
        let hidden = prefill[34].outputs["hidden"] ?? []
        let bulk = prefill[35]
        let fuses = Array(prefill[36..<68])
        let residual = prefill[68]
        #expect(Self.floats(bulk.inputs["hidden"]) == hidden)
        for (index, fuse) in fuses.enumerated() {
            let embed = encodes[index].outputs["curr_embed"] ?? []
            #expect(Self.floats(fuse.inputs["lm_raw"]) == Self.row(hidden, index + 1), "\(index)")
            #expect(Self.floats(fuse.inputs["curr_embed"]) == embed && embed != Self.row(hidden, index + 1), "\(index)")
            #expect(fuse.outputs["lm_hidden"] != fuse.outputs["residual_input"], "\(index)")
        }
        let bulkRows = bulk.outputs["residual_inputs"] ?? []
        let input = Self.floats(residual.inputs["embeds"])
        try #require(input.count == 145 * 2_048)
        for row in 0..<145 {
            let audio = (1...32).contains(row)
            let expected = audio ? fuses[row - 1].outputs["residual_input"] ?? [] : Self.row(bulkRows, row)
            #expect(Self.row(input, row) == expected && audio == (expected != Self.row(bulkRows, row)), "\(row)")
        }
        #expect(Self.integers(residual.inputs["length"]) == [Int32(Self.length)])
    }

    @Test(
        "VRP-02 Generation starts at row L - 1 with a zero condition and position L and decodes only generated patches"
    )
    func generatedPatches() async throws {
        let log = FakeVoxCPMLog()
        let speech = try await Self.pipeline(log).generate(
            ids: Self.body + [101], seed: 1708, reference: Self.voice(seconds: 5))
        let (prefill, loop) = try Self.phases(log.calls)
        try #require(prefill.map(\.function) == Self.prefillOrder(patches: 32))
        let last = Self.length - 1
        let heads = loop.filter { $0.function == .heads }
        #expect(Self.floats(heads.first?.inputs["lm_hidden"]) == Self.row(prefill[34].outputs["hidden"] ?? [], last))
        #expect(
            Self.floats(heads.first?.inputs["residual_hidden"]) == Self.row(prefill[68].outputs["hidden"] ?? [], last))
        let estimates = loop.filter { $0.function == .estimate }
        let generated = loop.filter { $0.function == .encode }
        let decodes = loop.filter { $0.function == .decode }
        try #require(heads.count == 4 && estimates.count == 36 && generated.count == 4 && decodes.count == 4)
        var previous = [Float](repeating: 0, count: VoxCPMSolver.patchSize)
        for (index, decode) in decodes.enumerated() {
            let latent = Self.floats(decode.inputs["latent"])
            #expect(latent == Self.floats(generated[index].inputs["patch"]), "\(index)")
            for estimate in estimates[(index * 9)..<((index + 1) * 9)] {
                #expect(Self.floats(estimate.inputs["cond"]) == previous + previous, "\(index)")
            }
            previous = latent
        }
        let positions = (0..<3).map { [Int32(Self.length + $0)] }
        #expect(loop.filter { $0.function == .step }.map { Self.integers($0.inputs["position"]) } == positions)
        #expect(loop.filter { $0.function == .residualStep }.map { Self.integers($0.inputs["position"]) } == positions)
        #expect(speech.samples == decodes.flatMap { $0.outputs["audio"] ?? [] })
        #expect(speech.samples.count == 4 * 7_680 && speech.patchCount == 4 && speech.stoppedPatch == 3)
    }

    @Test("VRP-02 The observer sees only generated patches and its replacement reaches both encoders and the condition")
    func observerReplacement() async throws {
        let log = FakeVoxCPMLog()
        let replacements = (0..<4).map { [Float](repeating: Float($0 + 1) / 32, count: VoxCPMSolver.patchSize) }
        let recorder = VoxCPMPatchRecorder(replacements: replacements)
        let speech = try await Self.pipeline(log).generate(
            ids: Self.body + [101], seed: 1708, reference: Self.voice(seconds: 5), observer: recorder.observe)
        let zero = [Float](repeating: 0, count: VoxCPMSolver.patchSize)
        #expect(recorder.patches.map(\.index) == [0, 1, 2, 3])
        #expect(recorder.patches.map(\.condition) == [zero] + replacements.prefix(3))
        let (_, loop) = try Self.phases(log.calls)
        #expect(loop.filter { $0.function == .encode }.map { Self.floats($0.inputs["patch"]) } == replacements)
        #expect(loop.filter { $0.function == .decode }.map { Self.floats($0.inputs["latent"]) } == replacements)
        #expect(speech.patchCount == 4)
    }

    @Test(
        "VRP-01 A five-second reference fits 110 target tokens, a ten-second one 79 and no reference 144 in 145 rows",
        arguments: [(5, 110, Int32(11)), (10, 79, Int32(73_447)), (0, 144, Int32(11))])
    func fullPrompt(seconds: Int, tokens: Int, token: Int32) async throws {
        let reference = seconds == 0 ? nil : try Self.voice(seconds: seconds)
        let ids = [Int32](repeating: token, count: tokens) + [101]
        let log = FakeVoxCPMLog()
        let speech = try await Self.pipeline(log).generate(ids: ids, seed: 3, reference: reference)
        #expect(speech.patchCount == 4)
        let calls = log.calls
        let padded = Self.integers(calls.first { $0.function == .embed }?.inputs["ids"])
        #expect(padded.count == 145 && padded.suffix(ids.count) == ids[...])
        #expect(Self.integers(calls.first { $0.function == .prefill }?.inputs["length"]) == [145])
        #expect(Self.integers(calls.first { $0.function == .step }?.inputs["position"]) == [145])
        let rejected = FakeVoxCPMLog()
        await #expect(throws: VoxCPMGenerationError.textTooLong(tokens + 1)) {
            try await Self.pipeline(rejected).generate(ids: [token] + ids, seed: 3, reference: reference)
        }
        #expect(rejected.events.isEmpty)
    }

    @Test(
        "VRP-01 Unsupported bundles, other formats, invalid IDs and early cancellation stop before a state or part call"
    )
    func rejection() async throws {
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(log)
        let legacy = try Self.pipeline(log, contract: VoxCPMPipelineTests.contract(.fp32))
        let voice = try Self.voice(seconds: 5)
        let irodori = try SpeechReferenceAudio(samples: [Float](repeating: 0.25, count: 24_000), source: .recording)
        for reference in [voice, irodori] {
            await #expect(throws: SpeechError.referenceNotSupported) {
                try await legacy.generate(ids: [8, 101], seed: 1, reference: reference)
            }
        }
        await #expect(throws: SpeechReferenceError.incompatibleFormat) {
            try await pipeline.generate(ids: [8, 101], seed: 1, reference: irodori)
        }
        let invalid: [[Int32]] = [[], [101], [8, 0], [-1, 101], [73_448, 101], [Int32.max, 101], [Int32.min, 101]]
        let references: [SpeechReferenceAudio?] = [voice, nil]
        for ids in invalid {
            for reference in references {
                await #expect(throws: VoxCPMGenerationError.invalidTokenIDs, "\(ids)") {
                    try await pipeline.generate(ids: ids, seed: 1, reference: reference)
                }
            }
        }
        let early = await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await pipeline.generate(ids: [8, 101], seed: 1, reference: voice)
        }.result
        #expect(VoxCPMPipelineTests.outcome(early) == "cancelled")
        #expect(log.events.isEmpty)
    }

    @Test("VRP-03 Three target tokens with a reference reach the text budget of 28 patches and return no audio")
    func textBudget() async throws {
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(log, [FakeVoxCPMBehavior(stopPatch: nil)])
        let voice = try Self.voice(seconds: 5)
        let result = await Task {
            try await pipeline.generate(ids: [0, 103, 104, 101], seed: 1, reference: voice)
        }.result
        #expect(VoxCPMPipelineTests.outcome(result) == "reachedPatchLimit(28)")
        #expect(log.calls.filter { $0.function == .decode }.count == 28)
        #expect(Self.lifecycle(log) == [.created(1), .released(1)])
    }

    @Test(
        "VRP-03 The cache budget subtracts the whole prompt length while the text budget counts only target tokens",
        arguments: [(5, 100, 25), (10, 79, 15), (0, 100, 59)])
    func cacheBudget(seconds: Int, tokens: Int, limit: Int) async throws {
        let log = FakeVoxCPMLog()
        let contract = try Self.capacityContract(160)
        let pipeline = try Self.pipeline(log, [FakeVoxCPMBehavior(stopPatch: nil)], contract: contract)
        let reference = seconds == 0 ? nil : try Self.voice(seconds: seconds)
        let ids = [Int32](repeating: 11, count: tokens) + [101]
        let result = await Task { try await pipeline.generate(ids: ids, seed: 1, reference: reference) }.result
        #expect(VoxCPMPipelineTests.outcome(result) == "reachedPatchLimit(\(limit))")
        #expect(log.calls.filter { $0.function == .decode }.count == limit)
    }

    @Test(
        "VRP-03 A missing, short or non-finite reference output fails at that call and the next generation succeeds",
        arguments: VoxCPMReferenceConditioningTests.damages)
    func damagedOutput(damage: Damage, outcome: String) async throws {
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(log, damage: damage)
        let voice = try Self.voice(seconds: 5)
        let failed = await Task { try await pipeline.generate(ids: Self.body + [101], seed: 1, reference: voice) }
            .result
        #expect(VoxCPMPipelineTests.outcome(failed) == outcome)
        let calls = log.calls(in: 1)
        #expect(calls.last?.function == damage.function)
        #expect(calls.filter { $0.function == damage.function }.count == damage.occurrence + 1)
        #expect(Self.lifecycle(log) == [.created(1), .released(1)])
        let speech = try await pipeline.generate(ids: Self.body + [101], seed: 1, reference: voice)
        #expect(speech.patchCount == 4)
        #expect(Self.lifecycle(log) == [.created(1), .released(1), .created(2), .released(2)])
    }

    @Test(
        "VRP-03 At each reference prefill call cancellation wins over a value or error and other errors pass unchanged",
        arguments: VoxCPMReferenceConditioningTests.prefillCalls)
    func interruption(function: VoxCPMFunction, occurrence: Int) async throws {
        let order = Self.prefillOrder(patches: 32)
        let index = try #require(order.indices.filter { order[$0] == function }.dropFirst(occurrence).first)
        let behaviors = [
            FakeVoxCPMBehavior(cancelledCall: index), FakeVoxCPMBehavior(failedCall: index),
            FakeVoxCPMBehavior(cancelledCall: index, failedCall: index), FakeVoxCPMBehavior(),
        ]
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(log, behaviors)
        let voice = try Self.voice(seconds: 5)
        for (generation, expected) in zip(1..., ["cancelled", "invalidOutput(\"fake\")", "cancelled"]) {
            let result = await Task {
                try await pipeline.generate(ids: Self.body + [101], seed: 1, reference: voice)
            }.result
            #expect(VoxCPMPipelineTests.outcome(result) == expected, "\(generation)")
            #expect(log.calls(in: generation).map(\.function) == Array(order.prefix(index + 1)), "\(generation)")
        }
        let speech = try await pipeline.generate(ids: Self.body + [101], seed: 1, reference: voice)
        #expect(speech.patchCount == 4)
        #expect(log.calls(in: 4).first { $0.function == .prefill }?.stateSum == 0)
        #expect(Self.lifecycle(log) == (1...4).flatMap { [FakeVoxCPMLog.Event.created($0), .released($0)] })
    }

    @Test(
        "VRP-03 Reference, plain and other reference generations alternate with new zeroed states and no carried values"
    )
    func alternatingGenerations() async throws {
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(log)
        let first = try Self.voice(seconds: 5)
        let second = try Self.voice(seconds: 10)
        let order: [SpeechReferenceAudio?] = [first, nil, second, first]
        for reference in order {
            let speech = try await pipeline.generate(ids: Self.body + [101], seed: 1708, reference: reference)
            #expect(speech.patchCount == 4)
        }
        #expect(
            Self.lifecycle(log) == [
                .created(1), .released(1), .created(2), .released(2), .created(3), .released(3), .created(4),
                .released(4),
            ])
        for (generation, reference) in zip(1..., order) {
            let calls = log.calls(in: generation)
            #expect(calls.filter { $0.function == .referenceEncode }.count == (reference == nil ? 0 : 1))
            let encodes = (reference?.conditioningPositionCount ?? 0) + 4
            #expect(calls.filter { $0.function == .encode }.count == encodes, "\(generation)")
            for function in [VoxCPMFunction.prefill, .residualPrefill, .decode] {
                #expect(calls.first { $0.function == function }?.stateSum == 0, "\(generation) \(function)")
            }
            #expect(calls.first { $0.function == .step }?.stateSum == 1, "\(generation)")
        }
        #expect(log.calls(in: 4).map(\.inputs) == log.calls(in: 1).map(\.inputs))
    }

    @Test(
        "VRP-03 A non-finite output of any of the twelve functions fails a reference generation at its first call",
        arguments: VoxCPMFunction.allCases)
    func nonFiniteOutput(_ function: VoxCPMFunction) async throws {
        let log = FakeVoxCPMLog()
        let pipeline = try Self.pipeline(log, [FakeVoxCPMBehavior(stopPatch: 3, nonFinite: function)])
        let voice = try Self.voice(seconds: 5)
        let result = await Task { try await pipeline.generate(ids: Self.body + [101], seed: 1, reference: voice) }
            .result
        let outputs = pipeline.contract.manifest.assets[function.asset]?.functions[function.name]?.outputs ?? []
        let name = try #require(outputs.map(\.name).sorted().first)
        #expect(VoxCPMPipelineTests.outcome(result) == "nonFiniteValue(\"\(function.rawValue).\(name)\")")
        #expect(log.calls.last?.function == function)
        #expect(log.calls.filter { $0.function == function }.count == 1)
    }

    @Test(
        "VRP-03 Without a reference both bundle formats keep the eleven-function calls of the earlier signatures",
        arguments: [VoxCPMBundleProfile.fp32, .referenceFP32])
    func plainGeneration(profile: VoxCPMBundleProfile) async throws {
        let contract = try VoxCPMPipelineTests.contract(profile)
        let original = FakeVoxCPMLog()
        let routed = FakeVoxCPMLog()
        let ids: [Int32] = [11, 12, 13, 101]
        let before = try await VoxCPMPipeline(
            components: FakeVoxCPMComponents(log: original, contract: contract), contract: contract
        ).generate(ids: ids, seed: 7)
        let after = try await VoxCPMPipeline(
            components: FakeVoxCPMComponents(log: routed, contract: contract), contract: contract
        ).generate(ids: ids, seed: 7, reference: nil)
        #expect(original.calls.map(\.function) == VoxCPMPipelineTests.sequence(patches: 4))
        #expect(routed.calls.map(\.function) == original.calls.map(\.function))
        #expect(routed.calls.map(\.inputs) == original.calls.map(\.inputs))
        #expect(after.samples == before.samples && after.patchCount == before.patchCount)
    }
}

extension VoxCPMReferenceConditioningTests {
    typealias Call = FakeVoxCPMLog.Call
    typealias Damage = FakeVoxCPMComponents.Damage

    static let body: [Int32] = [0, 103, 104, 101, 7]
    static let length = 1 + 32 + 1 + body.count + 1
    static let damages: [(Damage, String)] = [
        (Damage(function: .referenceEncode, output: "mu", kind: .missing), "invalidOutput(\"refenc.main.mu\")"),
        (Damage(function: .referenceEncode, output: "mu", kind: .short), "invalidOutput(\"refenc.main.mu\")"),
        (Damage(function: .referenceEncode, output: "mu", kind: .nonFiniteTail), "nonFiniteValue(\"refenc.main.mu\")"),
        (Damage(function: .encode, output: "curr_embed", kind: .short), "invalidOutput(\"locenc.main.curr_embed\")"),
        (
            Damage(function: .encode, occurrence: 31, output: "curr_embed", kind: .missing),
            "invalidOutput(\"locenc.main.curr_embed\")"
        ),
        (
            Damage(function: .fuse, output: "residual_input", kind: .short),
            "invalidOutput(\"proj.fuse.residual_input\")"
        ),
        (
            Damage(function: .fuse, occurrence: 31, output: "residual_input", kind: .missing),
            "invalidOutput(\"proj.fuse.residual_input\")"
        ),
    ]
    static let prefillCalls: [(VoxCPMFunction, Int)] = [
        (.referenceEncode, 0), (.encode, 0), (.encode, 31), (.prefill, 0), (.prefillFuse, 0), (.fuse, 0), (.fuse, 31),
        (.residualPrefill, 0),
    ]

    static func pipeline(
        _ log: FakeVoxCPMLog, _ behaviors: [FakeVoxCPMBehavior] = [FakeVoxCPMBehavior()], damage: Damage? = nil,
        contract: VoxCPMContract? = nil
    ) throws -> VoxCPMPipeline<FakeVoxCPMComponents> {
        let contract = try contract ?? VoxCPMPipelineTests.contract(.referenceFP32)
        let parts = FakeVoxCPMComponents(
            log: log, contract: contract, behaviors: behaviors, rowTag: 4_096, damage: damage)
        return VoxCPMPipeline(components: parts, contract: contract)
    }

    static func voice(seconds: Int) throws -> SpeechReferenceAudio {
        let samples = (0..<(seconds * 16_000)).map { Float($0 % 16_384 + 1) / 16_384 }
        return try SpeechReferenceAudio(samples: samples, source: .recording, format: .voxcpm2)
    }

    static func prefillOrder(patches: Int) -> [VoxCPMFunction] {
        let audio = [VoxCPMFunction](repeating: .encode, count: patches)
        let fused = [VoxCPMFunction](repeating: .fuse, count: patches)
        return [.referenceEncode] + audio + [.embed, .prefill, .prefillFuse] + fused + [.residualPrefill]
    }

    static func floats(_ input: VoxCPMInput?) -> [Float] { FakeVoxCPMComponents.floats(input) }
    static func integers(_ input: VoxCPMInput?) -> [Int32] { FakeVoxCPMComponents.integers(input) }

    static func row(_ values: [Float], _ index: Int) -> [Float] {
        let width = VoxCPMSolver.hiddenSize
        return Array(values[(index * width)..<((index + 1) * width)])
    }

    static func lifecycle(_ log: FakeVoxCPMLog) -> [FakeVoxCPMLog.Event] {
        log.events.filter { if case .called = $0 { false } else { true } }
    }

    static func phases(_ calls: [Call]) throws -> (prefill: [Call], loop: [Call]) {
        let end = try #require(calls.firstIndex { $0.function == .residualPrefill })
        return (Array(calls[...end]), Array(calls[(end + 1)...]))
    }

    static func capacityContract(_ capacity: Int) throws -> VoxCPMContract {
        let data = try VoxCPMBundleFixture.manifest(.referenceFP32)
        var json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        var execution = try #require(json["execution"] as? [String: Any])
        execution["capacity"] = capacity
        json["execution"] = execution
        let changed = try JSONSerialization.data(withJSONObject: json)
        return try VoxCPMContract(manifest: VoxCPMBundleInventory.manifest(in: changed))
    }
}
