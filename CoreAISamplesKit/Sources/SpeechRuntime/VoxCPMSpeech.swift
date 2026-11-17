#if os(macOS) || os(iOS)
    struct VoxCPMSpeech: Sendable {
        let samples: [Float]
        let sampleRate: Int
        let patchCount: Int
        let stoppedPatch: Int
    }
#endif
