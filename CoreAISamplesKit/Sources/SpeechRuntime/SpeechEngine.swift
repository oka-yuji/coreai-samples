#if os(macOS) || os(iOS)
    import SpeechCore

    public enum SpeechEngine: Sendable {
        case irodori
        case voxcpm2

        public var modelPathKey: String {
            switch self {
            case .irodori: "irodoriModelPath"
            case .voxcpm2: "voxcpm2ModelPath"
            }
        }

        var referenceFormat: SpeechReferenceFormat {
            switch self {
            case .irodori: .irodori
            case .voxcpm2: .voxcpm2
            }
        }

        func makeRuntime() -> any SpeechGenerating {
            switch self {
            case .irodori: IrodoriRuntime()
            case .voxcpm2: VoxCPMRuntime()
            }
        }
    }
#endif
