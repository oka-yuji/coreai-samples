#if os(macOS) || os(iOS)
    struct VoxCPMLoadStatistics: Sendable {
        let openSeconds: Double
        let assetSeconds: [String: Double]
        let totalSeconds: Double
    }
#endif
