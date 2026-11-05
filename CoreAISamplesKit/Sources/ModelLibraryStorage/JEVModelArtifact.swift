import CryptoKit
import Foundation
import ModelLibraryFeature

public enum JEVModelArtifact {
    public static let id = "jev-27b-coreai-uint4"
    public static let contract = "jev-27b-coreai-uint4-v1"
    public static let revision = "acc4cc835f64a75ab9ecee9d5d2e119a3bd29212"
    public static let manifestSHA256 = "2f34243a3d28500075e09d02d9f9f565a7c4b02f94a7d47b2a3caba7262d357d"

    public static func make() throws -> ModelArtifact {
        guard let url = Bundle.module.url(forResource: "jev-mac-files", withExtension: "json") else {
            throw ManagedModelStoreError.invalidDefinition
        }
        let files = try JSONDecoder().decode([ModelArtifactFile].self, from: Data(contentsOf: url))
        return ModelArtifact(
            id: id, title: "uint4 + fp32 head", contract: contract, platforms: [.macOS],
            sizeBytes: files.reduce(0) { $0 + $1.bytes },
            download: ModelDownload(repository: "okayuji/jev-27b-coreai", revision: revision, root: "bundle"),
            upstream: ModelUpstream(
                repository: "autotrust/JEV-27B", revision: "14b425269ed24452e0a5ad5a8f3e51d139647a88"),
            files: files)
    }

    static func validateManifest(_ data: Data) throws {
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard digest == manifestSHA256 else { throw ManagedModelStoreError.contractMismatch("JEV decision.json") }
    }
}
