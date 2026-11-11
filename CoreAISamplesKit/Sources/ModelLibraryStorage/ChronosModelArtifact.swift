import CryptoKit
import Foundation
import ModelLibraryFeature

public enum ChronosModelArtifact {
    public static let contract = "chronos2-fp32-v1"
    public static let modelID = "amazon/chronos-2"
    public static let revision = "29ec3766d36d6f73f0696f85560a422f50e8498c"
    public static let manifestSHA256 = "786dcf97e020c987cbf883f09b486fba5f3fee30a588b87a346ea65d72d20572"

    public static func make() throws -> ModelArtifact {
        guard let url = Bundle.module.url(forResource: "chronos2-fp32-files", withExtension: "json") else {
            throw ManagedModelStoreError.invalidDefinition
        }
        let files = try JSONDecoder().decode([ModelArtifactFile].self, from: Data(contentsOf: url))
        return ModelArtifact(
            id: "chronos2-fp32", title: "FP32", contract: contract, platforms: [.macOS, .iOS],
            sizeBytes: files.reduce(0) { $0 + $1.bytes },
            upstream: ModelUpstream(repository: modelID, revision: revision), files: files,
            localBundle: ModelLocalBundle(
                manifestSHA256: manifestSHA256, profileID: "chronos2-fp32", manifestPath: "chronos2.json"))
    }

    static func validateManifest(_ data: Data) throws {
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard hash == manifestSHA256 else { throw ManagedModelStoreError.contractMismatch("Chronos metadata") }
    }
}
