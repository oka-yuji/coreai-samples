#if os(macOS) || os(iOS)
    protocol VoxCPMSpeechPipeline: Sendable {
        @concurrent func generate(ids: [Int32], seed: UInt64) async throws -> VoxCPMSpeech
    }
#endif
