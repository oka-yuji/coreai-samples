enum VoxCPMGenerationDefect: CaseIterable, Sendable {
    case inferenceTimesteps, cfgValue, swaySamplingCoef, cfgZeroStar, zeroInitSteps, ditCallsPerPatch, minLen
    case maxPatchesRatio, maxPatchesOffset, maxLen, samplesPerPatch
    case audioStartToken, maxTextTokens, capacity, padToken, stateInitialValue, specialization

    var change: (path: [String], value: Any) {
        switch self {
        case .inferenceTimesteps: (["generation", "inference_timesteps"], 9)
        case .cfgValue: (["generation", "cfg_value"], 2.5)
        case .swaySamplingCoef: (["generation", "sway_sampling_coef"], 0.5)
        case .cfgZeroStar: (["generation", "cfg_zero_star"], false)
        case .zeroInitSteps: (["generation", "zero_init_steps"], 2)
        case .ditCallsPerPatch: (["generation", "dit_calls_per_patch"], 10)
        case .minLen: (["generation", "min_len"], 3)
        case .maxPatchesRatio: (["generation", "max_patches_ratio"], 5.5)
        case .maxPatchesOffset: (["generation", "max_patches_offset"], 11)
        case .maxLen: (["generation", "max_len"], 512)
        case .samplesPerPatch: (["generation", "samples_per_patch"], 0)
        case .audioStartToken: (["execution", "audio_start_token"], 102)
        case .maxTextTokens: (["execution", "max_text_tokens"], 100)
        case .capacity: (["execution", "capacity"], 145)
        case .padToken: (["execution", "pad_token"], -1)
        case .stateInitialValue: (["execution", "state_initial_value"], 1.0)
        case .specialization: (["execution", "specialization"], "cpu_only")
        }
    }
}
