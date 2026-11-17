#if os(macOS) || os(iOS)
    struct VoxCPMPatchRecord: Sendable {
        let index: Int
        let noise: [Float]
        let ditHidden: [Float]
        let condition: [Float]
        let latent: [Float]
        let stopLogits: [Float]
    }
#endif
