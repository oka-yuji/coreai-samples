#if os(macOS) || os(iOS)
    import SpeechCore
    import VoxCPMCore

    extension VoxCPMPipeline {
        static var vocabulary: ClosedRange<Int32> { 0...73_447 }

        func checkedPrompt(ids: [Int32], reference: SpeechReferenceAudio?) throws -> VoxCPMReferencePrompt {
            if let reference {
                guard contract.manifest.reference != nil else { throw SpeechError.referenceNotSupported }
                guard reference.format == .voxcpm2 else { throw SpeechReferenceError.incompatibleFormat }
            }
            guard ids.count > 1, ids.last == VoxCPMPrompt.audioStartTokenID,
                ids.allSatisfy({ Self.vocabulary.contains($0) })
            else { throw VoxCPMGenerationError.invalidTokenIDs }
            do {
                return try VoxCPMReferencePrompt(
                    textAndStartTokenIDs: ids, referencePatchCount: reference?.conditioningPositionCount ?? 0,
                    prefillLength: contract.prefillLength)
            } catch VoxCPMError.textTooLong {
                throw VoxCPMGenerationError.textTooLong(ids.count - 1)
            }
        }

        func prefill(
            _ prompt: VoxCPMReferencePrompt, reference: SpeechReferenceAudio?, _ states: inout Components.States
        ) async throws -> (lm: [Float], residual: [Float]) {
            let rows = prompt.audioPositions
            let acoustic = try await acousticEmbeddings(reference, count: rows.count, &states)
            let length = prompt.inputIDs.count
            let count = VoxCPMInput.integers([Int32(length)])
            let padded = prompt.inputIDs + [Int32](repeating: contract.padToken, count: contract.prefillLength - length)
            var embeds = try await run(.embed, ["ids": .integers(padded)], &states, "embeds")
            for (row, embed) in zip(rows, acoustic) { Self.replace(row, in: &embeds, with: embed) }
            let hidden = try await run(.prefill, ["embeds": .floats(embeds), "length": count], &states, "hidden")
            var residual = try await run(.prefillFuse, ["hidden": .floats(hidden)], &states, "residual_inputs")
            for (row, embed) in zip(rows, acoustic) {
                let inputs: [String: VoxCPMInput] = [
                    "lm_raw": .floats(Self.row(hidden, row)), "curr_embed": .floats(embed),
                ]
                let fused = try await run(.fuse, inputs, &states, "residual_input")
                Self.replace(row, in: &residual, with: fused)
            }
            let residualHidden = try await run(
                .residualPrefill, ["embeds": .floats(residual), "length": count], &states, "hidden")
            return (Self.row(hidden, length - 1), Self.row(residualHidden, length - 1))
        }

        private func acousticEmbeddings(
            _ reference: SpeechReferenceAudio?, count: Int, _ states: inout Components.States
        ) async throws -> [[Float]] {
            guard let reference, count > 0 else { return [] }
            let mu = try await run(.referenceEncode, ["pcm": .floats(reference.encoderSamples)], &states, "mu")
            let channels = VoxCPMContract.referenceOutput.shape[1]
            let frames = VoxCPMContract.referenceOutput.shape[2]
            let width = VoxCPMSolver.patchSize / channels
            var embeds: [[Float]] = []
            for patch in 0..<count {
                let values = (0..<VoxCPMSolver.patchSize).map { mu[$0 / width * frames + patch * width + $0 % width] }
                let embed = try await run(.encode, ["patch": .floats(values)], &states, "curr_embed")
                embeds.append(embed)
            }
            return embeds
        }

        private static func replace(_ row: Int, in values: inout [Float], with replacement: [Float]) {
            let width = VoxCPMSolver.hiddenSize
            values.replaceSubrange((row * width)..<((row + 1) * width), with: replacement)
        }
    }
#endif
