#if os(macOS) || os(iOS)
    import CoreAI

    struct VoxCPMAssetStates: Sendable {
        var textKey: NDArray
        var textValue: NDArray
        var residualKey: NDArray
        var residualValue: NDArray
        var decoder: VoxCPMDecoderStates
    }
#endif
