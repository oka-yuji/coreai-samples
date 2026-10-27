import ModelLibraryFeature
import SpeechAssets

public enum IrodoriModelArtifact {
    public static let contract = "irodori-coreai-v1"

    public static func make() throws -> ModelArtifact {
        let files = try IrodoriBundleInventory.files().map {
            ModelArtifactFile(path: $0.path, bytes: $0.bytes, sha256: $0.sha256)
        }
        return ModelArtifact(
            id: "irodori-v4.1-small-mf-fp32", title: "FP32", contract: contract, platforms: [.iOS],
            sizeBytes: files.reduce(0) { $0 + $1.bytes }, files: files,
            localBundle: ModelLocalBundle(
                manifestSHA256: IrodoriBundleInventory.manifestSHA256, profileID: "irodori-v4.1-small-mf-fp32",
                manifestPath: "irodori.json"))
    }
}
