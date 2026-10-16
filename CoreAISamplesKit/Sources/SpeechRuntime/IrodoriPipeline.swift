#if os(macOS) || os(iOS)
    import CoreAI
    import Foundation
    import IrodoriInputLogic
    import SpeechCore

    struct Context: SpeechContext {
        let text: NDArray
        let mask: NDArray
        let frames: Int
    }

    struct Pipeline: SpeechPipeline {
        let stages: [Stage]
        init(assets: [URL]) async throws {
            guard assets.count == 4 else { throw SpeechError.invalidAsset("Expected four model components") }
            var values: [Stage] = []
            for (index, asset) in assets.enumerated() {
                values.append(try await Stage(url: asset, dynamicShape: index >= 2))
            }
            stages = values
        }
        @concurrent
        func prepare(_ input: IrodoriTextInput) async throws -> any SpeechContext {
            let ids = NDArray(scalars: input.tokenIDs, shape: [1, 256])
            let mask = NDArray(scalars: input.attentionMask, shape: [1, 256])
            let text = try await stages[0].run(["input_ids": ids, "attention_mask": mask])
            let result = try await stages[1].run(["text_state": text, "text_mask": mask])
            guard let logFrames = try floats(result).first else { throw SpeechError.invalidDuration }
            return Context(text: text, mask: mask, frames: try resolveFrames(logFrames: logFrames, fixed: nil))
        }
        @concurrent
        func generate(_ value: any SpeechContext, seed: UInt64) async throws -> [Float] {
            guard let context = value as? Context else { throw SpeechError.invalidSamples }
            var latent = try IrodoriNoise.generate(frames: context.frames, seed: seed)
            let speaker = NDArray(scalars: [Float](repeating: 0, count: 1536), shape: [1, 2, 768])
            let speakerMask = NDArray(scalars: [Bool](repeating: false, count: 2), shape: [1, 2])
            let caption = NDArray(scalars: [Float](repeating: 0, count: 256 * 512), shape: [1, 256, 512])
            let captionMask = NDArray(scalars: [Bool](repeating: false, count: 256), shape: [1, 256])
            for step in 0..<4 {
                let velocity = try await stages[2].run([
                    "latent": NDArray(scalars: latent, shape: [1, context.frames, 32]),
                    "t": NDArray(scalars: [Float(1) - Float(step) * 0.25], shape: [1]),
                    "delta_t": NDArray(scalars: [Float(0.25)], shape: [1]),
                    "text_state": context.text, "text_mask": context.mask,
                    "speaker_state": speaker, "speaker_mask": speakerMask,
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
