#if os(macOS) || os(iOS)
    import SpeechCore

    extension VoxCPMPipeline: VoxCPMSpeechPipeline {
        @concurrent
        func generate(ids: [Int32], seed: UInt64, reference: SpeechReferenceAudio?) async throws -> VoxCPMSpeech {
            try await generate(ids: ids, seed: seed, reference: reference, observer: nil)
        }
    }
#endif
