import Foundation
import ModelLibraryFeature
import SpeechAssets

public enum VoxCPMModelArtifact {
    public static let profile = VoxCPMBundleProfile.fp16
    public static let contract = "voxcpm2-coreai-v1"

    public static func make() throws -> ModelArtifact {
        try make(files: VoxCPMBundleInventory.files(profile: profile))
    }

    static func make(inventory: Data) throws -> ModelArtifact {
        try make(files: VoxCPMBundleInventory.files(inventory: inventory))
    }

    static func validateManifest(_ data: Data) throws {
        try require(VoxCPMBundleInventory.profile(in: data))
    }

    static func verifyConfiguration(of source: URL) throws {
        guard let data = try? readLocalBundleFile(source: source, path: VoxCPMBundleInventory.manifestPath),
            let declared = try? VoxCPMBundleInventory.profile(in: data)
        else { return }
        try require(declared)
    }

    private static func require(_ declared: VoxCPMBundleProfile) throws {
        guard declared == profile else {
            throw ManagedModelStoreError.contractMismatch("VoxCPM2 configuration " + declared.rawValue)
        }
    }

    private static func make(files: [IrodoriBundleInventory.File]) throws -> ModelArtifact {
        guard let manifest = files.first(where: { $0.path == VoxCPMBundleInventory.manifestPath }) else {
            throw ManagedModelStoreError.invalidDefinition
        }
        let id = "voxcpm2-" + profile.rawValue
        let entries = files.map { ModelArtifactFile(path: $0.path, bytes: $0.bytes, sha256: $0.sha256) }
        return ModelArtifact(
            id: id, title: title, contract: contract, platforms: [.macOS],
            sizeBytes: entries.reduce(0) { $0 + $1.bytes },
            upstream: ModelUpstream(
                repository: VoxCPMBundleInventory.modelID, revision: VoxCPMBundleInventory.revision),
            files: entries,
            localBundle: ModelLocalBundle(
                manifestSHA256: manifest.sha256, profileID: id, manifestPath: VoxCPMBundleInventory.manifestPath))
    }

    private static var title: String {
        switch profile {
        case .fp32: "FP32"
        case .fp16: "FP16"
        case .fp16Proj32: "FP16 (FP32 projection)"
        }
    }
}
