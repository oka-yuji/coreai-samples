#if os(macOS) || os(iOS)
    import CoreAI
    import Foundation
    import IrodoriInputLogic
    import SpeechCore

    struct Context: SpeechContext {
        let text: NDArray
        let mask: NDArray
        let frames: Int
        let speaker: NDArray
        let speakerMask: NDArray
    }

    struct Pipeline: SpeechPipeline {
        let stages: [Stage]
        init(assets: [URL]) async throws {
            guard [4, 6].contains(assets.count) else {
                throw SpeechError.invalidAsset("Expected four or six model components")
            }
            var values: [Stage] = []
            for (index, asset) in assets.enumerated() {
                let referenceStage = assets.count == 6 && [1, 4, 5].contains(index)
                #if os(iOS)
                    let phoneDecoder = index == 3
                #else
                    let phoneDecoder = false
                #endif
                values.append(
                    try await Stage(
                        url: asset, dynamicShape: index >= 2 || referenceStage,
                        frequentReshapes: referenceStage || phoneDecoder ? false : nil))
            }
            stages = values
        }
        @concurrent
        func prepare(_ input: IrodoriTextInput) async throws -> any SpeechContext {
            try await prepare(input, reference: nil)
        }
        @concurrent
        func prepare(_ input: IrodoriTextInput, reference: SpeechReferenceAudio?) async throws -> any SpeechContext {
            guard reference == nil || stages.count == 6 else { throw SpeechError.referenceNotSupported }
            let ids = NDArray(scalars: input.tokenIDs, shape: [1, 256])
            let mask = NDArray(scalars: input.attentionMask, shape: [1, 256])
            let text = try await stages[0].run(["input_ids": ids, "attention_mask": mask])
            let voice = try await speaker(reference)
            var durationInputs = ["text_state": text, "text_mask": mask]
            if stages.count == 6 {
                durationInputs["speaker_state"] = voice.state
                durationInputs["speaker_mask"] = voice.mask
            }
            let result = try await stages[1].run(durationInputs)
            guard let logFrames = try floats(result).first else { throw SpeechError.invalidDuration }
            return Context(
                text: text, mask: mask, frames: try resolveFrames(logFrames: logFrames, fixed: nil),
                speaker: voice.state, speakerMask: voice.mask)
        }

        @concurrent
        private func speaker(_ reference: SpeechReferenceAudio?) async throws -> (state: NDArray, mask: NDArray) {
            guard let reference else {
                return (
                    NDArray(scalars: [Float](repeating: 0, count: 1536), shape: [1, 2, 768]),
                    NDArray(scalars: [Bool](repeating: false, count: 2), shape: [1, 2])
                )
            }
            let audio = NDArray(scalars: reference.encoderSamples, shape: [1, 1, reference.frameCount * 1920])
            let encoded = try await stages[4].run(["audio": audio])
            let samples = try floats(encoded)
            guard encoded.shape == [1, reference.frameCount, 32], samples.allSatisfy({ $0.isFinite }) else {
                throw SpeechError.invalidSamples
            }
            let usable = reference.frameCount / 4 * 4
            let latent = NDArray(scalars: Array(samples.prefix(usable * 32)), shape: [1, usable, 32])
            let mask = NDArray(scalars: [Bool](repeating: true, count: usable), shape: [1, usable])
            let state = try await stages[5].run(["latent": latent, "mask": mask])
            guard state.shape == [1, reference.speakerTokenCount, 768], try floats(state).allSatisfy({ $0.isFinite })
            else {
                throw SpeechError.invalidSamples
            }
            return (
                state,
                NDArray(
                    scalars: [Bool](repeating: true, count: reference.speakerTokenCount),
                    shape: [1, reference.speakerTokenCount])
            )
        }
        @concurrent
        func generate(_ value: any SpeechContext, seed: UInt64) async throws -> [Float] {
            guard let context = value as? Context else { throw SpeechError.invalidSamples }
            var latent = try IrodoriNoise.generate(frames: context.frames, seed: seed)
            let caption = NDArray(scalars: [Float](repeating: 0, count: 256 * 512), shape: [1, 256, 512])
            let captionMask = NDArray(scalars: [Bool](repeating: false, count: 256), shape: [1, 256])
            for step in 0..<4 {
                let velocity = try await stages[2].run([
                    "latent": NDArray(scalars: latent, shape: [1, context.frames, 32]),
                    "t": NDArray(scalars: [Float(1) - Float(step) * 0.25], shape: [1]),
                    "delta_t": NDArray(scalars: [Float(0.25)], shape: [1]),
                    "text_state": context.text, "text_mask": context.mask,
                    "speaker_state": context.speaker, "speaker_mask": context.speakerMask,
                    "caption_state": caption, "caption_mask": captionMask,
                ])
                let update = try floats(velocity)
                guard update.count == latent.count else { throw SpeechError.invalidSamples }
                for index in latent.indices { latent[index] -= update[index] * 0.25 }
            }
            let decoded = try await stages[3].run(["latent": NDArray(scalars: latent, shape: [1, context.frames, 32])])
            let pcm = try floats(decoded)
            guard pcm.count == context.frames * 1920, pcm.allSatisfy({ $0.isFinite }) else {
                throw SpeechError.invalidSamples
            }
            return pcm
        }
    }

#endif
