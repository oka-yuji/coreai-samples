import Foundation
import SpeechCore

public struct VoxCPMBundleManifest: Decodable, Sendable {
    public struct Model: Decodable, Sendable {
        public let modelID: String
        public let revision: String
        public let upstreamRevision: String
        public let license: String

        enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case revision
            case upstreamRevision = "upstream_revision"
            case license
        }
    }

    public struct Configuration: Decodable, Sendable {
        public let name: String
        public let precisions: [String: String]
    }

    public struct Tensor: Decodable, Equatable, Sendable {
        public let name: String
        public let dtype: String
        public let shape: [Int]

        public init(from decoder: any Decoder) throws {
            var container = try decoder.unkeyedContainer()
            name = try container.decode(String.self)
            dtype = try container.decode(String.self)
            shape = try container.decode([Int].self)
            guard container.isAtEnd else {
                throw DecodingError.dataCorruptedError(in: container, debugDescription: "A tensor has three fields.")
            }
        }
    }

    public struct Function: Decodable, Sendable {
        public let inputs: [Tensor]
        public let outputs: [Tensor]
        public let states: [Tensor]
    }

    public struct Asset: Decodable, Sendable {
        public let path: String
        public let precision: String
        public let functions: [String: Function]
    }

    public struct Execution: Decodable, Sendable {
        public let capacity: Int
        public let prefillLength: Int
        public let maxTextTokens: Int
        public let audioStartToken: Int32
        public let padToken: Int32
        public let stateInitialValue: Double
        public let specialization: String

        enum CodingKeys: String, CodingKey {
            case capacity
            case prefillLength = "prefill_length"
            case maxTextTokens = "max_text_tokens"
            case audioStartToken = "audio_start_token"
            case padToken = "pad_token"
            case stateInitialValue = "state_initial_value"
            case specialization
        }
    }

    public struct Generation: Decodable, Sendable {
        public let inferenceTimesteps: Int
        public let cfgValue: Double
        public let swaySamplingCoef: Double
        public let cfgZeroStar: Bool
        public let zeroInitSteps: Int
        public let ditCallsPerPatch: Int
        public let minLen: Int
        public let maxPatchesRatio: Double
        public let maxPatchesOffset: Int
        public let maxLen: Int
        public let samplesPerPatch: Int

        enum CodingKeys: String, CodingKey {
            case inferenceTimesteps = "inference_timesteps"
            case cfgValue = "cfg_value"
            case swaySamplingCoef = "sway_sampling_coef"
            case cfgZeroStar = "cfg_zero_star"
            case zeroInitSteps = "zero_init_steps"
            case ditCallsPerPatch = "dit_calls_per_patch"
            case minLen = "min_len"
            case maxPatchesRatio = "max_patches_ratio"
            case maxPatchesOffset = "max_patches_offset"
            case maxLen = "max_len"
            case samplesPerPatch = "samples_per_patch"
        }
    }

    public struct Structure: Decodable, Sendable {
        public let patch: [Int]
        public let sampleRate: Int

        enum CodingKeys: String, CodingKey {
            case patch
            case sampleRate = "sample_rate"
        }
    }

    public let format: Int
    public let model: Model
    public let configuration: Configuration
    public let assets: [String: Asset]
    public let execution: Execution
    public let generation: Generation
    public let structure: Structure
    public let tokenizer: String
    public let reference: VoxCPMReferenceConfiguration?

    public func validate() throws -> VoxCPMBundleProfile {
        typealias Contract = VoxCPMBundleInventory
        guard VoxCPMBundleProfile.allCases.contains(where: { $0.format == format }) else {
            throw Self.invalid("has an unsupported format")
        }
        guard model.modelID == Contract.modelID, model.revision == Contract.revision,
            model.upstreamRevision == Contract.upstreamRevision
        else { throw Self.invalid("describes another model or revision") }
        guard let profile = VoxCPMBundleProfile(rawValue: configuration.name),
            configuration.precisions == profile.precisions
        else { throw Self.invalid("has an unsupported configuration") }
        guard format == profile.format else { throw Self.invalid("has another format than its configuration") }
        guard reference == profile.reference else {
            throw Self.invalid("has other reference audio settings than its configuration")
        }
        guard structure.sampleRate == Contract.sampleRate, structure.patch == Contract.patch else {
            throw Self.invalid("has another sample rate or patch shape")
        }
        guard tokenizer == Contract.tokenizerPath else { throw Self.invalid("names another tokenizer") }
        guard Set(assets.keys) == Set(profile.assetNames) else { throw Self.invalid("lists other assets") }
        for name in profile.assetNames {
            guard let asset = assets[name], let precision = profile.precisions[name], asset.precision == precision,
                asset.path == "artifacts/\(name)-\(precision).aimodel",
                Set(asset.functions.keys) == Contract.functions[name]
            else { throw Self.invalid("has an unexpected \(name) asset") }
        }
        return profile
    }

    private static func invalid(_ detail: String) -> SpeechError {
        .invalidAsset("The VoxCPM2 manifest " + detail + ".")
    }
}
