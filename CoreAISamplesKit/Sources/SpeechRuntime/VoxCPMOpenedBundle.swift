#if os(macOS) || os(iOS)
    struct VoxCPMOpenedBundle: Sendable {
        let bundle: ValidatedSpeechBundle
        let contract: VoxCPMContract

        var name: String {
            switch contract.profile {
            case .fp32: "VoxCPM2 (FP32)"
            case .fp16: "VoxCPM2 (FP16)"
            case .fp16Proj32: "VoxCPM2 (FP16, FP32 projection)"
            case .referenceFP32: "VoxCPM2 (FP32, reference audio)"
            }
        }
    }
#endif
