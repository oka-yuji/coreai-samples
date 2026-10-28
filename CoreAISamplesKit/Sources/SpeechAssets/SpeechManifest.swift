import Foundation
import SpeechCore

public enum SpeechBundleProfile: String, Sendable {
    case textOnly = "irodori-coreai-v1"
    case referenceVoice = "irodori-coreai-v2"
}

public struct IrodoriAssetSpec: Decodable {
    public struct FileSpec: Decodable {
        public let bytes: Int
        public let sha256: String
    }
    public let stem: String
    public let files: [String: FileSpec]
}

struct SpeechManifest: Codable {
    let format: String
    let modelID: String
    let precision: String
    let sampleRate: Int

    func validate() throws {
        guard SpeechBundleProfile(rawValue: format) != nil, modelID == IrodoriBundleInventory.modelID,
            precision == "fp32", sampleRate == 48000
        else {
            throw SpeechError.invalidAsset("Select a supported Irodori FP32 model folder.")
        }
    }
}
