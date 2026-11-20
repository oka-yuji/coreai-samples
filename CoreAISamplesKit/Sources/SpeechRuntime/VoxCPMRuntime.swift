#if os(macOS) || os(iOS)
    import Foundation
    import SpeechCore
    import VoxCPMCore

    actor VoxCPMRuntime: SpeechGenerating {
        private let dependencies: VoxCPMDependencies
        private var opened: VoxCPMOpenedBundle?
        private var tokenizer: (any VoxCPMTextTokenizer)?
        private var pipeline: (any VoxCPMSpeechPipeline)?
        private var busy = false
        private var releasing = false
        private var waiters: [CheckedContinuation<Void, Never>] = []
        private(set) var lastLoadSeconds: Double?

        init(dependencies: VoxCPMDependencies = .live) { self.dependencies = dependencies }

        func open(_ directory: URL) throws -> SpeechModel {
            guard !busy else { throw SpeechError.busy }
            try Task.checkCancellation()
            pipeline = nil
            tokenizer = nil
            opened = nil
            let bundle = try dependencies.open(directory)
            let input = try makeTokenizer(bundle)
            try Task.checkCancellation()
            opened = bundle
            tokenizer = input
            return SpeechModel(
                name: bundle.name, directory: directory,
                supportsReferenceAudio: bundle.contract.profile == .referenceFP32)
        }

        func generate(text: String, seed: UInt64) async throws -> SpeechAudio {
            try await synthesize(text: text, seed: seed, reference: nil)
        }

        func generate(text: String, seed: UInt64, reference: SpeechReferenceAudio?) async throws -> SpeechAudio {
            try await synthesize(text: text, seed: seed, reference: reference)
        }

        func release() async {
            if busy {
                releasing = true
                dependencies.didRequestRelease()
                await withCheckedContinuation { waiters.append($0) }
            } else {
                pipeline = nil
                tokenizer = nil
            }
        }

        private func synthesize(text: String, seed: UInt64, reference: SpeechReferenceAudio?) async throws
            -> SpeechAudio
        {
            guard !busy else { throw SpeechError.busy }
            try Task.checkCancellation()
            guard let opened else { throw SpeechError.modelNotOpened("VoxCPM2") }
            let request = try checkedRequest(text: text, seed: seed, reference: reference, opened: opened)
            busy = true
            defer { finishOperation() }
            let start = ContinuousClock.now
            lastLoadSeconds = nil
            do {
                if pipeline == nil {
                    pipeline = try await dependencies.pipeline(opened)
                    lastLoadSeconds = (ContinuousClock.now - start) / .seconds(1)
                }
                try Task.checkCancellation()
                let speech = try await speech(request)
                try Task.checkCancellation()
                let wav = try request.wav(speech, samplesPerPatch: opened.contract.samplesPerPatch)
                return SpeechAudio(
                    wav: wav, audioSeconds: Double(speech.samples.count) / 48_000,
                    elapsedSeconds: (ContinuousClock.now - start) / .seconds(1),
                    normalizedText: request.normalizedText)
            } catch {
                pipeline = nil
                if Task.isCancelled { throw CancellationError() }
                switch error {
                case VoxCPMGenerationError.reachedPatchLimit: throw SpeechError.speechNotFinished
                case is VoxCPMGenerationError, is VoxCPMError: throw SpeechError.invalidSamples
                default: throw error
                }
            }
        }

        private func checkedRequest(
            text: String, seed: UInt64, reference: SpeechReferenceAudio?, opened: VoxCPMOpenedBundle
        ) throws -> VoxCPMGenerationRequest {
            let normalized = try VoxCPMGenerationRequest.checkedText(
                text, reference: reference, profile: opened.contract.profile)
            let input = try tokenizer ?? makeTokenizer(opened)
            tokenizer = input
            return try VoxCPMGenerationRequest(
                normalizedText: normalized, seed: seed, reference: reference, contract: opened.contract,
                tokenizer: input)
        }

        private func makeTokenizer(_ bundle: VoxCPMOpenedBundle) throws -> any VoxCPMTextTokenizer {
            do { return try dependencies.tokenizer(bundle.bundle.tokenizer) } catch {
                throw SpeechError.invalidAsset("The VoxCPM2 tokenizer could not be loaded.")
            }
        }

        private func speech(_ request: VoxCPMGenerationRequest) async throws -> VoxCPMSpeech {
            guard let pipeline else { throw SpeechError.modelNotOpened("VoxCPM2") }
            return try await pipeline.generate(ids: request.ids, seed: request.seed, reference: request.reference)
        }

        private func finishOperation() {
            busy = false
            if releasing {
                pipeline = nil
                tokenizer = nil
                releasing = false
            }
            let pending = waiters
            waiters.removeAll()
            for waiter in pending { waiter.resume() }
        }
    }
#endif
