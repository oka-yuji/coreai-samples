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
            return SpeechModel(name: bundle.name, directory: directory)
        }

        func generate(text: String, seed: UInt64) async throws -> SpeechAudio {
            guard !busy else { throw SpeechError.busy }
            try Task.checkCancellation()
            guard let opened else { throw SpeechError.modelNotOpened("VoxCPM2") }
            let normalized: String
            do { normalized = try VoxCPMText.normalized(text) } catch { throw SpeechError.emptyText }
            if tokenizer == nil { tokenizer = try makeTokenizer(opened) }
            guard let tokenizer else { throw SpeechError.modelNotOpened("VoxCPM2") }
            let ids: [Int32]
            do {
                ids = try VoxCPMPrompt.inputIDs(
                    textTokenIDs: tokenizer.tokenIDs(for: normalized), vocabulary: tokenizer.vocabulary)
            } catch { throw SpeechError.unreadableText }
            let contract = opened.contract
            let limit = contract.prefillLength - 1
            guard ids.count > 1 else { throw SpeechError.emptyText }
            guard ids.count - 1 <= limit,
                let patches = try? VoxCPMStopRule.patchLimit(textTokenCount: ids.count - 1, capacity: contract.capacity)
            else { throw SpeechError.textExceedsLimit(count: ids.count - 1, limit: limit) }
            busy = true
            defer { finishOperation() }
            let start = ContinuousClock.now
            lastLoadSeconds = nil
            do {
                if pipeline == nil {
                    pipeline = try await dependencies.pipeline(opened)
                    lastLoadSeconds = (ContinuousClock.now - start) / .seconds(1)
                }
                let speech = try await speech(ids, seed: seed)
                try Task.checkCancellation()
                guard speech.sampleRate == 48_000, speech.samples.count == speech.patchCount * contract.samplesPerPatch
                else { throw SpeechError.invalidSamples }
                let wav = try floatWAV(samples: speech.samples, maximumSamples: patches * contract.samplesPerPatch)
                return SpeechAudio(
                    wav: wav, audioSeconds: Double(speech.samples.count) / 48_000,
                    elapsedSeconds: (ContinuousClock.now - start) / .seconds(1), normalizedText: normalized)
            } catch {
                pipeline = nil
                switch error {
                case VoxCPMGenerationError.reachedPatchLimit: throw SpeechError.speechNotFinished
                case is VoxCPMGenerationError, is VoxCPMError: throw SpeechError.invalidSamples
                default: throw error
                }
            }
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

        private func makeTokenizer(_ bundle: VoxCPMOpenedBundle) throws -> any VoxCPMTextTokenizer {
            do { return try dependencies.tokenizer(bundle.bundle.tokenizer) } catch {
                throw SpeechError.invalidAsset("The VoxCPM2 tokenizer could not be loaded.")
            }
        }

        private func speech(_ ids: [Int32], seed: UInt64) async throws -> VoxCPMSpeech {
            guard let pipeline else { throw SpeechError.modelNotOpened("VoxCPM2") }
            return try await pipeline.generate(ids: ids, seed: seed)
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
