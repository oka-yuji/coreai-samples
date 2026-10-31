import Foundation
import SpeechCore

public enum IrodoriBundleInventory {
    public struct File: Equatable, Sendable {
        public let path: String
        public let bytes: Int64
        public let sha256: String
    }

    public static let manifestSHA256 = "ce41c61dd2249cab61aa69017316627c54329177fbfb19a1d1df4d775d65dbc7"
    public static let referenceManifestSHA256 = "964e4c6f3d5f235fff5bac346d545a928112274412241d348ab3c65ff5f449a5"
    public static let modelID = "Aratako/Irodori-TTS-v4.1-Small-MF"

    public static func files(profile: SpeechBundleProfile = .textOnly) throws -> [File] {
        let specs = try assetSpecifications(profile: profile)
        let assets = specs.flatMap { spec in
            spec.files.map { name, file in
                File(path: "artifacts/" + spec.stem + ".aimodel/" + name, bytes: Int64(file.bytes), sha256: file.sha256)
            }
        }
        return
            (assets + [
                File(
                    path: "irodori.json", bytes: profile == .textOnly ? 132 : 226, sha256: manifestSHA256(for: profile)),
                File(
                    path: "tokenizer.json", bytes: 6_718_495,
                    sha256: "6a0734cf21c802169defaffe719bc2ef12bb9d0be37e54b61ed27aa89394723d"),
            ]).sorted { $0.path < $1.path }
    }

    public static func assetSpecifications(profile: SpeechBundleProfile = .textOnly) throws -> [IrodoriAssetSpec] {
        let name = profile == .textOnly ? "irodori-assets" : "irodori-reference-assets"
        guard let rules = Bundle.module.url(forResource: name, withExtension: "json") else {
            throw SpeechError.invalidAsset("The application's Irodori asset manifest is missing.")
        }
        return try JSONDecoder().decode([IrodoriAssetSpec].self, from: Data(contentsOf: rules))
    }

    public static func validateManifest(_ data: Data) throws {
        try JSONDecoder().decode(SpeechManifest.self, from: data).validate()
    }

    public static func profile(in data: Data) throws -> SpeechBundleProfile {
        let manifest = try JSONDecoder().decode(SpeechManifest.self, from: data)
        try manifest.validate()
        guard let profile = SpeechBundleProfile(rawValue: manifest.format) else {
            throw SpeechError.invalidAsset("Unsupported Irodori model profile.")
        }
        return profile
    }

    public static func manifestSHA256(for profile: SpeechBundleProfile) -> String {
        profile == .textOnly ? manifestSHA256 : referenceManifestSHA256
    }
}
