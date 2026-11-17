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
        let stateSum = group.map(states.sum)
        if behavior.cancelledCall == call { withUnsafeCurrentTask { $0?.cancel() } }
        var result = outputs(function, inputs, occurrence: occurrence, stopPatch: behavior.stopPatch)
        if behavior.nonFinite == function, behavior.nonFiniteOccurrence == occurrence,
            let name = result.keys.sorted().first
        {
            result[name]?[0] = .nan
        }
        log.record(
            FakeVoxCPMLog.Call(
                generation: states.generation, function: function, inputs: inputs, outputs: result,
                stateSum: stateSum))
        if behavior.failedCall == call { throw VoxCPMGenerationError.invalidOutput("fake") }
        if let group { states.write(group, 1) }
        return result
    }

    private func outputs(_ function: VoxCPMFunction, _ inputs: [String: VoxCPMInput], occurrence: Int, stopPatch: Int?)
        -> [String: [Float]]
    {
        let sizes = Dictionary(
            uniqueKeysWithValues: (contract.manifest.assets[function.asset]?.functions[function.name]?.outputs ?? [])
                .map { ($0.name, $0.shape.reduce(1, *)) })
        func values(_ name: String) -> [Float] { Self.floats(inputs[name]) }
        func spread(_ source: [Float], _ name: String, _ transform: (Float) -> Float) -> [Float] {
            source.isEmpty ? [] : (0..<(sizes[name] ?? 0)).map { transform(source[$0 % source.count]) }
        }
        let position = Float(Self.integers(inputs["position"]).first ?? -1)
        switch function {
        case .embed:
            let ids = Self.integers(inputs["ids"])
            let width = (sizes["embeds"] ?? 0) / max(ids.count, 1)
            return ["embeds": ids.flatMap { id in (0..<width).map { Float(id) + Float($0 % 5) * 0.25 } }]
        case .prefill:
            return ["hidden": values("embeds").map { $0 * 2 + 1 }]
        case .prefillFuse:
            return ["residual_inputs": values("hidden").map { $0 * 3 - 1 }]
        case .residualPrefill:
            return ["hidden": values("embeds").map { $0 * 0.5 + 7 }]
        case .heads:
            let stop = stopPatch.map { occurrence >= $0 } ?? false
            return [
                "dit_hidden": zip(values("lm_hidden"), values("residual_hidden")).map { $0 + 2 * $1 },
                "stop_logits": stop ? [0, 1] : [1, 0],
            ]
        case .estimate:
            let x = values("x")
            let mu = values("mu")
            let cond = values("cond")
            let half = max(x.count / 2, 1)
            let width = mu.count / 2
            return [
                "velocity": x.indices.map { index in
                    let hidden = index / half * width + index % half
                    return 0.01 * x[index] + 0.001 * (mu.indices.contains(hidden) ? mu[hidden] : 0)
                        + 0.0001 * (cond.indices.contains(index) ? cond[index] : 0)
                }
            ]
        case .encode:
            return ["curr_embed": spread(values("patch"), "curr_embed") { $0 * 0.5 + 0.125 }]
        case .decode:
            return ["audio": spread(values("latent"), "audio") { $0 * 0.1 }]
        case .step:
            return ["hidden": values("embed").map { $0 * 1.5 + position * 0.01 }]
        case .fuse:
            let raw = values("lm_raw")
            let embed = values("curr_embed")
            return ["lm_hidden": zip(raw, embed).map { $0 + 0.25 * $1 }, "residual_input": zip(raw, embed).map(-)]
        case .residualStep:
            return ["hidden": values("embed").map { $0 * 0.75 + position * 0.02 }]
        }
    }
}
