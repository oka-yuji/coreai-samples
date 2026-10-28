#if os(macOS) || os(iOS)
    import Foundation
    import IrodoriInputLogic
    import SpeechCore

    public actor IrodoriRuntime: SpeechGenerating {
        private let dependencies: SpeechDependencies
        private var bundle: ValidatedSpeechBundle?
        private var tokenizer: (any SpeechTokenizer)?
        private var pipeline: (any SpeechPipeline)?
        private var policy = RecyclePolicy()
        private var previousReferenceFrames: Int?
        private var busy = false
        private var releasing = false
        private var waiters: [CheckedContinuation<Void, Never>] = []

        public init() { dependencies = .live }
        init(dependencies: SpeechDependencies) { self.dependencies = dependencies }

        public func open(_ directory: URL) throws -> SpeechModel {
            guard !busy, !releasing else { throw SpeechError.busy }
            try Task.checkCancellation()
            clearModels()
            bundle = nil
            tokenizer = nil
            let validated = try dependencies.validate(directory)
            let input = try dependencies.tokenizer(validated.tokenizer)
            try Task.checkCancellation()
            bundle = validated
            tokenizer = input
            return SpeechModel(
                name: "Irodori v4.1 Small MF (FP32)", directory: directory,
                supportsReferenceAudio: validated.assets.count == 6)
        }

        public func generate(text: String, seed: UInt64) async throws -> SpeechAudio {
            try await generate(text: text, seed: seed, reference: nil)
        }

        public func generate(text: String, seed: UInt64, reference: SpeechReferenceAudio?) async throws -> SpeechAudio {
            guard !busy, !releasing else { throw SpeechError.busy }
            try Task.checkCancellation()
            guard let bundle else { throw SpeechError.noModel }
            guard reference == nil || bundle.assets.count == 6 else { throw SpeechError.referenceNotSupported }
            if tokenizer == nil { tokenizer = try dependencies.tokenizer(bundle.tokenizer) }
            guard let tokenizer else { throw SpeechError.noModel }
            let input: IrodoriTextInput
            do { input = try tokenizer.prepare(text) } catch IrodoriInputError.emptyText { throw SpeechError.emptyText }
            guard input.originalTokenCount <= 255 else { throw SpeechError.textTooLong(input.originalTokenCount) }
            try Task.checkCancellation()
            busy = true
            defer { finishOperation() }
            let start = ContinuousClock.now
            var context: (any SpeechContext)?
            do {
                let referenceFrames = reference?.frameCount ?? 0
                if let previousReferenceFrames, previousReferenceFrames != referenceFrames { clearModels() }
                if pipeline == nil { pipeline = try await dependencies.pipeline(bundle.assets) }
                try Task.checkCancellation()
                context = try await prepare(input, reference: reference)
                guard let frames = context?.frames, (13...750).contains(frames) else {
                    throw SpeechError.invalidDuration
                }
                if policy.needsReload(for: frames) {
                    context = nil
                    pipeline = nil
                    policy.reset()
                    pipeline = try await dependencies.pipeline(bundle.assets)
                    try Task.checkCancellation()
                    context = try await prepare(input, reference: reference)
                    guard context?.frames == frames else { throw SpeechError.invalidDuration }
                }
                let pcm = try await samples(context, seed: seed)
                context = nil
                try Task.checkCancellation()
                guard pcm.count == frames * 1920 else { throw SpeechError.invalidSamples }
                let wav = try floatWAV(samples: pcm)
                try Task.checkCancellation()
                policy.recordCompletion(frames: frames)
                previousReferenceFrames = referenceFrames
                return SpeechAudio(
                    wav: wav, audioSeconds: Double(frames) * 0.04,
                    elapsedSeconds: (ContinuousClock.now - start) / .seconds(1),
                    normalizedText: input.normalizedText)
            } catch {
                context = nil
                clearModels()
                throw error
            }
        }

        public func release() async {
            if busy {
                releasing = true
                dependencies.didRequestRelease()
                await withCheckedContinuation { waiters.append($0) }
            } else {
                clearModels()
                tokenizer = nil
            }
        }

        private func prepare(_ input: IrodoriTextInput, reference: SpeechReferenceAudio?) async throws
            -> any SpeechContext
        {
            guard let pipeline else { throw SpeechError.noModel }
            return try await pipeline.prepare(input, reference: reference)
        }

        private func samples(_ context: (any SpeechContext)?, seed: UInt64) async throws -> [Float] {
            guard let pipeline, let context else { throw SpeechError.noModel }
            return try await pipeline.generate(context, seed: seed)
        }

        private func finishOperation() {
            busy = false
            if releasing {
                clearModels()
                tokenizer = nil
            }
            releasing = false
            let pending = waiters
            waiters.removeAll()
            for waiter in pending { waiter.resume() }
        }

        private func clearModels() {
            pipeline = nil
            policy.reset()
            previousReferenceFrames = nil
        }
    }
#endif
