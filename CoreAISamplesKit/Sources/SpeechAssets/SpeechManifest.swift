import Foundation
import SpeechCore

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
        guard format == "irodori-coreai-v1", modelID == IrodoriBundleInventory.modelID,
            precision == "fp32", sampleRate == 48000
        else {
            throw SpeechError.invalidAsset("Select a supported Irodori FP32 model folder.")
        }
    }
}
