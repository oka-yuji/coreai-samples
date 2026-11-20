#if os(macOS) || os(iOS)
    import Foundation
    import SpeechAssets
    import SpeechCore
    import VoxCPMCore

    struct VoxCPMGenerationRequest: Sendable {
        static let tokenIDs: ClosedRange<Int32> = 0...73_447

        let normalizedText: String
        let ids: [Int32]
        let seed: UInt64
        let reference: SpeechReferenceAudio?
        let patchLimit: Int

        static func checkedText(_ text: String, reference: SpeechReferenceAudio?, profile: VoxCPMBundleProfile)
            throws -> String
        {
            if let reference {
                guard profile == .referenceFP32 else { throw SpeechError.referenceNotSupported }
                guard reference.format == .voxcpm2 else { throw SpeechReferenceError.incompatibleFormat }
            }
            do { return try VoxCPMText.normalized(text) } catch { throw SpeechError.emptyText }
        }

        init(
            normalizedText: String, seed: UInt64, reference: SpeechReferenceAudio?, contract: VoxCPMContract,
            tokenizer: any VoxCPMTextTokenizer
        ) throws {
            let ids: [Int32]
            do {
                ids = try VoxCPMPrompt.inputIDs(
                    textTokenIDs: tokenizer.tokenIDs(for: normalizedText), vocabulary: tokenizer.vocabulary)
            } catch { throw SpeechError.unreadableText }
            guard ids.allSatisfy({ Self.tokenIDs.contains($0) }) else { throw SpeechError.unreadableText }
            let count = ids.count - 1
            guard count > 0 else { throw SpeechError.emptyText }
            let patches = reference?.conditioningPositionCount ?? 0
            let limit: Int
            do {
                let prompt = try VoxCPMReferencePrompt(
                    textAndStartTokenIDs: ids, referencePatchCount: patches, prefillLength: contract.prefillLength)
                limit = try VoxCPMStopRule.patchLimit(
                    prompt: prompt, capacity: contract.capacity, maximumPatches: contract.manifest.generation.maxLen)
            } catch {
                let prefix = patches > 0 ? patches + 2 : 0
                throw SpeechError.textExceedsLimit(count: count, limit: contract.prefillLength - prefix - 1)
            }
            self.normalizedText = normalizedText
            self.ids = ids
            self.seed = seed
            self.reference = reference
            patchLimit = limit
        }

        func wav(_ speech: VoxCPMSpeech, samplesPerPatch: Int) throws -> Data {
            let count = speech.patchCount
            guard speech.sampleRate == 48_000, count >= 1, count <= patchLimit, speech.stoppedPatch == count - 1,
                speech.samples.count == count * samplesPerPatch
            else { throw SpeechError.invalidSamples }
            return try floatWAV(samples: speech.samples, maximumSamples: patchLimit * samplesPerPatch)
        }
    }
#endif
