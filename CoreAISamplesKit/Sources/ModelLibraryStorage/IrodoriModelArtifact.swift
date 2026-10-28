import ModelLibraryFeature
import SpeechAssets

public enum IrodoriModelArtifact {
    public static let contract = "irodori-coreai-v1"
    public static let referenceContract = "irodori-coreai-v2"

    public static func make(profile: SpeechBundleProfile = .textOnly) throws -> ModelArtifact {
        let files = try IrodoriBundleInventory.files(profile: profile).map {
            ModelArtifactFile(path: $0.path, bytes: $0.bytes, sha256: $0.sha256)
        }
        let id = profile == .textOnly ? "irodori-v4.1-small-mf-fp32" : "irodori-v4.1-small-mf-reference-fp32"
        return ModelArtifact(
            id: id, title: profile == .textOnly ? "FP32" : "Reference voice (FP32)",
            contract: profile.rawValue, platforms: [.iOS],
            sizeBytes: files.reduce(0) { $0 + $1.bytes }, files: files,
            localBundle: ModelLocalBundle(
                manifestSHA256: IrodoriBundleInventory.manifestSHA256(for: profile), profileID: id,
                manifestPath: "irodori.json"))
    }
}
