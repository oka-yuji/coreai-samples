#if os(macOS) || os(iOS)
    import SpeechCore

    protocol VoxCPMSpeechPipeline: Sendable {
        @concurrent func generate(ids: [Int32], seed: UInt64) async throws -> VoxCPMSpeech
        @concurrent func generate(ids: [Int32], seed: UInt64, reference: SpeechReferenceAudio?) async throws
            -> VoxCPMSpeech
    }
#endif
