#if os(macOS) || os(iOS)
    import SpeechAssets
    import SpeechCore
    import VoxCPMCore

    struct VoxCPMContract: Sendable {
        typealias Tensor = VoxCPMSignature.Tensor

        static let specialization = "gpu_preferred"
        static let maximumCapacity = 8_192
        static let decoderStateNames = (0..<26).map { $0 < 10 ? "state_0\($0)" : "state_\($0)" }
        static let cacheStateNames = ["k_cache", "v_cache"]
        static let referenceInput = Tensor(
            name: "pcm", type: "float32", shape: [1, 1, VoxCPMReferenceConfiguration.paddedSamples])
        static let referenceOutput = Tensor(name: "mu", type: "float32", shape: [1, 64, 252])

        let manifest: VoxCPMBundleManifest
        let profile: VoxCPMBundleProfile

        var prefillLength: Int { manifest.execution.prefillLength }
        var capacity: Int { manifest.execution.capacity }
        var padToken: Int32 { manifest.execution.padToken }
        var samplesPerPatch: Int { manifest.generation.samplesPerPatch }
        var sampleRate: Int { manifest.structure.sampleRate }

        init(manifest: VoxCPMBundleManifest) throws {
            profile = try manifest.validate()
            let generation = manifest.generation
            let execution = manifest.execution
            guard generation.inferenceTimesteps == VoxCPMSolver.stepCount,
                generation.cfgValue == Double(VoxCPMSolver.guidanceScale),
                generation.swaySamplingCoef == Double(VoxCPMSolver.swaySamplingCoefficient),
                generation.cfgZeroStar == VoxCPMSolver.usesZeroStar,
                generation.zeroInitSteps == VoxCPMSolver.zeroInitSteps,
                generation.ditCallsPerPatch == VoxCPMSolver.estimatorCallsPerPatch,
                generation.minLen == VoxCPMStopRule.minimumLength,
                generation.maxPatchesRatio == Double(VoxCPMStopRule.patchesPerTextToken),
                generation.maxPatchesOffset == VoxCPMStopRule.extraPatches, generation.maxLen >= execution.capacity,
                generation.samplesPerPatch > 0
            else { throw Self.invalid("Its generation settings differ from the VoxCPM2 solver and stop rule.") }
            guard execution.audioStartToken == VoxCPMPrompt.audioStartTokenID, execution.prefillLength > 1,
                execution.maxTextTokens == execution.prefillLength - 1,
                execution.capacity > execution.prefillLength, execution.capacity <= Self.maximumCapacity,
                execution.padToken >= 0,
                execution.stateInitialValue == 0, execution.specialization == Self.specialization,
                manifest.structure.patch.reduce(1, *) == VoxCPMSolver.patchSize
            else { throw Self.invalid("Its execution settings differ from the VoxCPM2 pipeline.") }
            self.manifest = manifest
        }

        func outputCount(_ function: VoxCPMFunction, _ name: String) -> Int? {
            manifest.assets[function.asset]?.functions[function.name]?.outputs.first { $0.name == name }
                .flatMap { Self.count($0.shape) }
        }

        static func count(_ shape: [Int]) -> Int? {
            var count = 1
            for dimension in shape {
                let product = count.multipliedReportingOverflow(by: dimension)
                guard dimension > 0, !product.overflow else { return nil }
                count = product.partialValue
            }
            return count
        }

        func check(asset: String, functionNames: [String], observed: [String: VoxCPMSignature]) throws {
            guard let declared = manifest.assets[asset], functionNames.count == declared.functions.count,
                Set(functionNames) == Set(declared.functions.keys)
            else { throw Self.invalid("The \(asset) asset holds other functions than the manifest.") }
            let float = declared.precision == "fp16" ? "float16" : "float32"
            for (name, function) in declared.functions {
                guard let signature = observed[name], signature == VoxCPMSignature(function),
                    let kind = VoxCPMFunction(rawValue: asset + "." + name), expects(kind, signature, float)
                else { throw Self.invalid("The \(asset).\(name) descriptors differ from the contract.") }
            }
            if let prefill = observed["prefill"], let step = observed["step"], prefill.states != step.states {
                throw Self.invalid("The \(asset) prefill and step states differ.")
            }
        }

        static func checkSharedStates<Descriptor: Equatable>(asset: String, _ states: [String: [String: Descriptor]])
            throws
        {
            guard let prefill = states["prefill"], let step = states["step"] else { return }
            guard !prefill.isEmpty, prefill == step else {
                throw invalid("The \(asset) prefill and step functions do not share one state layout.")
            }
        }

        private func expects(_ function: VoxCPMFunction, _ signature: VoxCPMSignature, _ float: String) -> Bool {
            let hidden = VoxCPMSolver.hiddenSize
            let rows = prefillLength
            let patch = manifest.structure.patch
            let pair = [2] + patch.dropFirst()
            func value(_ name: String, _ shape: [Int]) -> Tensor { Tensor(name: name, type: float, shape: shape) }
            func index(_ name: String, _ shape: [Int]) -> Tensor { Tensor(name: name, type: "int32", shape: shape) }
            let expected: (inputs: [Tensor], outputs: [Tensor], states: [String])
            switch function {
            case .embed:
                expected = ([index("ids", [1, rows])], [value("embeds", [1, rows, hidden])], [])
            case .prefill, .residualPrefill:
                expected = (
                    [value("embeds", [1, rows, hidden]), index("length", [1])], [value("hidden", [1, rows, hidden])],
                    Self.cacheStateNames
                )
            case .step, .residualStep:
                expected = (
                    [value("embed", [1, hidden]), index("position", [1])], [value("hidden", [1, hidden])],
                    Self.cacheStateNames
                )
            case .heads:
                expected = (
                    [value("lm_hidden", [1, hidden]), value("residual_hidden", [1, hidden])],
                    [value("dit_hidden", [1, hidden]), value("stop_logits", [1, 2])], []
                )
            case .fuse:
                expected = (
                    [value("lm_raw", [1, hidden]), value("curr_embed", [1, hidden])],
                    [value("lm_hidden", [1, hidden]), value("residual_input", [1, hidden])], []
                )
            case .prefillFuse:
                expected = ([value("hidden", [1, rows, hidden])], [value("residual_inputs", [1, rows, hidden])], [])
            case .encode:
                expected = ([value("patch", patch)], [value("curr_embed", [1, hidden])], [])
            case .estimate:
                expected = (
                    [
                        value("x", pair), value("mu", [2, hidden]), value("t", [2]), value("cond", pair),
                        value("dt", [2]),
                    ],
                    [value("velocity", pair)], []
                )
            case .decode:
                expected = (
                    [value("latent", patch)], [value("audio", [1, 1, samplesPerPatch])], Self.decoderStateNames
                )
            case .referenceEncode:
                expected = ([Self.referenceInput], [Self.referenceOutput], [])
            }
            let caches = expected.states == Self.cacheStateNames
            return signature.inputs == expected.inputs && signature.outputs == expected.outputs
                && signature.states.map(\.name) == expected.states
                && signature.states.allSatisfy { state in
                    state.type == float && !state.shape.isEmpty && state.shape.allSatisfy { $0 > 0 }
                        && (!caches || (state.shape.count == 4 && state.shape[2] == capacity))
                }
        }

        private static func invalid(_ detail: String) -> SpeechError {
            .invalidAsset("The VoxCPM2 bundle cannot run. " + detail)
        }
    }
#endif
