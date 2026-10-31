import ModelLibraryFeature
import SpeechAssets

public enum IrodoriModelArtifact {
    public static let contract = "irodori-coreai-v1"
    public static let referenceContract = "irodori-coreai-v2"

    public static func makeReferenceDownload() throws -> ModelArtifact {
        let local = try make(profile: .referenceVoice)
        let files =
            (local.files + [
                ModelArtifactFile(
                    path: "LICENSE", bytes: 14_849,
                    sha256: "0a169976e646de2c797c944a8d5068c38aad412028e89cec5ba8660ff92e7576"),
                ModelArtifactFile(
                    path: "NOTICE", bytes: 2_032,
                    sha256: "eb36db717126e1152f207c2aced17f9027f2aa41756ac66a451e7a837e2a2672"),
            ]).sorted { $0.path < $1.path }
        return ModelArtifact(
            id: local.id, title: local.title, contract: local.contract, platforms: local.platforms,
            sizeBytes: files.reduce(0) { $0 + $1.bytes },
            download: ModelDownload(
                repository: "okayuji/irodori-v4.1-small-mf-coreai-fp32",
                revision: "f5db5096365c7d1388a5002cfb3280f66cb19375"),
            upstream: ModelUpstream(
                repository: "Aratako/Irodori-TTS-v4.1-Small-MF",
                revision: "ccc78f5d480b6e51b69b2d5042a14c4da04fea6e"), files: files)
    }

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
