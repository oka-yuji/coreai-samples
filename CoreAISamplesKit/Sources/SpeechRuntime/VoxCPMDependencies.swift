#if os(macOS) || os(iOS)
    import Foundation
    import VoxCPMTokenizers

    struct VoxCPMDependencies: Sendable {
        var open: @Sendable (URL) throws -> VoxCPMOpenedBundle
        var tokenizer: @Sendable (URL) throws -> any VoxCPMTextTokenizer
        var pipeline: @Sendable (VoxCPMOpenedBundle) async throws -> any VoxCPMSpeechPipeline
        var didRequestRelease: @Sendable () -> Void = {}

        static let live = VoxCPMDependencies(
            open: { try VoxCPMPipeline<VoxCPMAssets>.open(bundle: $0) },
            tokenizer: { try VoxCPMTokenizer(tokenizerJSON: $0) },
            pipeline: { try await VoxCPMPipeline<VoxCPMAssets>.load($0) })
    }
#endif
