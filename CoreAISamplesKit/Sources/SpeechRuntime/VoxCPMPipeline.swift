#if os(macOS) || os(iOS)
    import Foundation
    import SpeechCore
    import VoxCPMCore

    struct VoxCPMPipeline<Components: VoxCPMComponents>: Sendable {
        let components: Components
        let contract: VoxCPMContract

        @concurrent
        func generate(ids: [Int32], seed: UInt64) async throws -> VoxCPMSpeech {
            try await generate(ids: ids, seed: seed, observer: nil)
        }

        @concurrent
        func generate(
            ids: [Int32], seed: UInt64, observer: (@Sendable (VoxCPMPatchRecord) throws -> [Float]?)?
        ) async throws -> VoxCPMSpeech {
            try await generate(ids: ids, seed: seed, reference: nil, observer: observer)
        }

        @concurrent
        func generate(
            ids: [Int32], seed: UInt64, reference: SpeechReferenceAudio?,
            observer: (@Sendable (VoxCPMPatchRecord) throws -> [Float]?)? = nil
        ) async throws -> VoxCPMSpeech {
            try Task.checkCancellation()
            let prompt = try checkedPrompt(ids: ids, reference: reference)
            let limit = try VoxCPMStopRule.patchLimit(
                prompt: prompt, capacity: contract.capacity, maximumPatches: contract.manifest.generation.maxLen)
            let length = prompt.inputIDs.count
            var states = try components.makeStates()
            var (lmRow, residualRow) = try await prefill(prompt, reference: reference, &states)
            var condition = [Float](repeating: 0, count: VoxCPMSolver.patchSize)
            var noise = VoxCPMNoise(seed: seed)
            var samples: [Float] = []
            for index in 0..<limit {
                let heads = try await call(
                    .heads, ["lm_hidden": .floats(lmRow), "residual_hidden": .floats(residualRow)], &states)
                let ditHidden = try output(heads, .heads, "dit_hidden")
                let stopLogits = try output(heads, .heads, "stop_logits")
                let patchNoise = noise.nextPatch()
                var solver = try VoxCPMSolver(noise: patchNoise, ditHidden: ditHidden, condition: condition)
                while let input = solver.estimatorInput {
                    let inputs: [String: VoxCPMInput] = [
                        "x": .floats(input.x), "mu": .floats(input.mu), "t": .floats(input.t),
                        "cond": .floats(input.cond), "dt": .floats(input.dt),
                    ]
                    try solver.advance(velocity: try await run(.estimate, inputs, &states, "velocity"))
                }
                var latent = solver.latent
                let record = VoxCPMPatchRecord(
                    index: index, noise: patchNoise, ditHidden: ditHidden, condition: condition, latent: latent,
                    stopLogits: stopLogits)
                if let replacement = try observer?(record) {
                    guard replacement.count == latent.count else { throw VoxCPMError.invalidSize }
                    latent = replacement
                }
                let embed = try await run(.encode, ["patch": .floats(latent)], &states, "curr_embed")
                samples += try await run(.decode, ["latent": .floats(latent)], &states, "audio")
                switch try VoxCPMStopRule.step(patchIndex: index, limit: limit, stopLogits: stopLogits) {
                case .stopped:
                    return VoxCPMSpeech(
                        samples: samples, sampleRate: contract.sampleRate, patchCount: index + 1, stoppedPatch: index)
                case .reachedLimit:
                    throw VoxCPMGenerationError.reachedPatchLimit(limit)
                case .next:
                    break
                }
                let position = VoxCPMInput.integers([Int32(length + index)])
                let raw = try await run(.step, ["embed": .floats(embed), "position": position], &states, "hidden")
                let fused = try await call(.fuse, ["lm_raw": .floats(raw), "curr_embed": .floats(embed)], &states)
                lmRow = try output(fused, .fuse, "lm_hidden")
                let residualInput = try output(fused, .fuse, "residual_input")
                residualRow = try await run(
                    .residualStep, ["embed": .floats(residualInput), "position": position], &states, "hidden")
                condition = latent
            }
            throw VoxCPMGenerationError.reachedPatchLimit(limit)
        }

        func run(
            _ function: VoxCPMFunction, _ inputs: [String: VoxCPMInput], _ states: inout Components.States,
            _ name: String
        ) async throws -> [Float] {
            try output(try await call(function, inputs, &states), function, name)
        }

        private func call(
            _ function: VoxCPMFunction, _ inputs: [String: VoxCPMInput], _ states: inout Components.States
        ) async throws -> [String: [Float]] {
            try Task.checkCancellation()
            let outputs: [String: [Float]]
            do {
                outputs = try await components.call(function, inputs, states: &states)
            } catch {
                try Task.checkCancellation()
                throw error
            }
            try Task.checkCancellation()
            for name in outputs.keys.sorted() where outputs[name]?.allSatisfy(\.isFinite) == false {
                throw VoxCPMGenerationError.nonFiniteValue(function.rawValue + "." + name)
            }
            return outputs
        }

        private func output(_ outputs: [String: [Float]], _ function: VoxCPMFunction, _ name: String) throws -> [Float]
        {
            guard let values = outputs[name], values.count == contract.outputCount(function, name) else {
                throw VoxCPMGenerationError.invalidOutput(function.rawValue + "." + name)
            }
            return values
        }

        static func row(_ values: [Float], _ index: Int) -> [Float] {
            let width = VoxCPMSolver.hiddenSize
            return Array(values[(index * width)..<((index + 1) * width)])
        }
    }
#endif
