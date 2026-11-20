#if os(macOS)
    import Foundation
    import SpeechCore

    @testable import SpeechRuntime

    actor VoxCPMPrefixTrace {
        private let folder: URL
        private var prefill = true
        private var firstHead = true
        private var firstEstimate = true

        init(folder: URL) throws {
            self.folder = folder
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        }

        struct Parts: VoxCPMComponents {
            let physical: VoxCPMAssets
            let trace: VoxCPMPrefixTrace

            func makeStates() throws -> VoxCPMAssetStates { try physical.makeStates() }

            @concurrent
            func call(_ function: VoxCPMFunction, _ inputs: [String: VoxCPMInput], states: inout VoxCPMAssetStates)
                async throws -> [String: [Float]]
            {
                let outputs = try await physical.call(function, inputs, states: &states)
                try await trace.record(function.rawValue, inputs, outputs)
                return outputs
            }
        }

        private func append<T>(_ values: [T], _ path: String) throws {
            let url = folder.appending(path: path)
            let data = values.withUnsafeBytes { Data($0) }
            if !FileManager.default.fileExists(atPath: url.path) {
                try data.write(to: url, options: .withoutOverwriting)
                return
            }
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            _ = try handle.seekToEnd()
            try handle.write(contentsOf: data)
        }

        private func record(_ function: String, _ input: [String: VoxCPMInput], _ output: [String: [Float]]) throws {
            var name = function
            var inputs = [String]()
            var outputs = [String]()
            switch function {
            case "refenc.main":
                inputs = ["pcm"]
                outputs = ["mu"]
            case "tslm.embed":
                inputs = ["ids"]
                outputs = ["embeds"]
            case "locenc.main":
                name = prefill ? "reference-locenc" : "generated-locenc"
                inputs = ["patch"]
                if prefill { outputs = ["curr_embed"] }
            case "tslm.prefill", "ralm.prefill":
                inputs = ["embeds", "length"]
                outputs = ["hidden"]
                if function == "ralm.prefill" { prefill = false }
            case "proj.prefill_fuse":
                inputs = ["hidden"]
                outputs = ["residual_inputs"]
            case "proj.fuse" where prefill:
                name = "reference-fuse"
                inputs = ["lm_raw", "curr_embed"]
                outputs = ["residual_input"]
            case "proj.heads":
                outputs = ["stop_logits"]
                if firstHead {
                    inputs = ["lm_hidden", "residual_hidden"]
                    firstHead = false
                }
            case "dit.main" where firstEstimate:
                inputs = ["cond"]
                firstEstimate = false
            case "tslm.step", "ralm.step":
                inputs = ["position"]
            case "vae.main":
                inputs = ["latent"]
            default: break
            }
            for key in inputs {
                let path = name + ".in." + key
                switch input[key] {
                case .floats(let x): try append(x, path + ".f32")
                case .integers(let x): try append(x, path + ".i32")
                case nil: throw SpeechError.invalidSamples
                }
            }
            for key in outputs {
                guard let x = output[key] else { throw SpeechError.invalidSamples }
                try append(x, name + ".out." + key + ".f32")
            }
        }
    }
#endif
