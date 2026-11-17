import Foundation

@testable import SpeechRuntime

struct FakeVoxCPMComponents: VoxCPMComponents {
    let log: FakeVoxCPMLog
    let contract: VoxCPMContract
    var behaviors = [FakeVoxCPMBehavior()]

    static func group(_ function: VoxCPMFunction) -> String? {
        switch function {
        case .prefill, .step: "tslm"
        case .residualPrefill, .residualStep: "ralm"
        case .decode: "vae"
        default: nil
        }
    }

    static func floats(_ input: VoxCPMInput?) -> [Float] {
        if case .floats(let values) = input { return values }
        return []
    }

    static func integers(_ input: VoxCPMInput?) -> [Int32] {
        if case .integers(let values) = input { return values }
        return []
    }

    func makeStates() throws -> FakeVoxCPMStates {
        FakeVoxCPMStates(log: log)
    }

    @concurrent
    func call(_ function: VoxCPMFunction, _ inputs: [String: VoxCPMInput], states: inout FakeVoxCPMStates)
        async throws -> [String: [Float]]
    {
        let (call, occurrence) = states.next(function)
        let behavior = behaviors[min(states.generation, behaviors.count) - 1]
        let group = Self.group(function)
        log.record(
            FakeVoxCPMLog.Call(
                generation: states.generation, function: function, inputs: inputs, stateSum: group.map(states.sum)))
        if behavior.cancelledCall == call { withUnsafeCurrentTask { $0?.cancel() } }
        var result = outputs(function, inputs, occurrence: occurrence, stopPatch: behavior.stopPatch)
        if let group { states.write(group, 1) }
        if behavior.nonFinite == function, behavior.nonFiniteOccurrence == occurrence,
            let name = result.keys.sorted().first
        {
            result[name]?[0] = .nan
        }
        return result
    }

    private func outputs(_ function: VoxCPMFunction, _ inputs: [String: VoxCPMInput], occurrence: Int, stopPatch: Int?)
        -> [String: [Float]]
    {
        let sizes = Dictionary(
            uniqueKeysWithValues: (contract.manifest.assets[function.asset]?.functions[function.name]?.outputs ?? [])
                .map { ($0.name, $0.shape.reduce(1, *)) })
        func fill(_ name: String, _ value: Float) -> [Float] {
            [Float](repeating: value, count: sizes[name] ?? 0)
        }
        let position = Float(Self.integers(inputs["position"]).first ?? -1)
        switch function {
        case .embed:
            let ids = Self.integers(inputs["ids"])
            let width = (sizes["embeds"] ?? 0) / max(ids.count, 1)
            return ["embeds": ids.flatMap { [Float](repeating: Float($0), count: width) }]
        case .prefill:
            let rows = contract.prefillLength
            let width = (sizes["hidden"] ?? 0) / rows
            return ["hidden": (0..<rows).flatMap { [Float](repeating: Float($0) + 0.5, count: width) }]
        case .prefillFuse:
            return ["residual_inputs": Self.floats(inputs["hidden"]).map { $0 + 1000 }]
        case .residualPrefill:
            return ["hidden": Self.floats(inputs["embeds"]).map { $0 + 0.25 }]
        case .heads:
            let stop = stopPatch.map { occurrence >= $0 } ?? false
            return [
                "dit_hidden": zip(Self.floats(inputs["lm_hidden"]), Self.floats(inputs["residual_hidden"])).map(+),
                "stop_logits": stop ? [0, 1] : [1, 0],
            ]
        case .estimate:
            return ["velocity": fill("velocity", 0)]
        case .encode:
            return ["curr_embed": fill("curr_embed", Float(occurrence) + 0.125)]
        case .decode:
            return ["audio": fill("audio", Float(occurrence + 1) / 1000)]
        case .step:
            return ["hidden": fill("hidden", position)]
        case .fuse:
            return ["lm_hidden": Self.floats(inputs["lm_raw"]), "residual_input": Self.floats(inputs["curr_embed"])]
        case .residualStep:
            return ["hidden": fill("hidden", position + 0.5)]
        }
    }
}
