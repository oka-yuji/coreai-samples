import Foundation
import SpeechCore

public enum IrodoriBundleInventory {
    public struct File: Equatable, Sendable {
        public let path: String
        public let bytes: Int64
        public let sha256: String
    }

    public static let manifestSHA256 = "ce41c61dd2249cab61aa69017316627c54329177fbfb19a1d1df4d775d65dbc7"
    public static let modelID = "Aratako/Irodori-TTS-v4.1-Small-MF"

    public static func files() throws -> [File] {
        let specs = try assetSpecifications()
        let assets = specs.flatMap { spec in
            spec.files.map { name, file in
                File(path: "artifacts/" + spec.stem + ".aimodel/" + name, bytes: Int64(file.bytes), sha256: file.sha256)
            }
        }
        return
            (assets + [
                File(path: "irodori.json", bytes: 132, sha256: manifestSHA256),
                File(
                    path: "tokenizer.json", bytes: 6_718_495,
                    sha256: "6a0734cf21c802169defaffe719bc2ef12bb9d0be37e54b61ed27aa89394723d"),
            ]).sorted { $0.path < $1.path }
    }

    public static func assetSpecifications() throws -> [IrodoriAssetSpec] {
        guard let rules = Bundle.module.url(forResource: "irodori-assets", withExtension: "json") else {
            throw SpeechError.invalidAsset("The application's Irodori asset manifest is missing.")
        }
        return try JSONDecoder().decode([IrodoriAssetSpec].self, from: Data(contentsOf: rules))
    }

    public static func validateManifest(_ data: Data) throws {
        try JSONDecoder().decode(SpeechManifest.self, from: data).validate()
    }
}
