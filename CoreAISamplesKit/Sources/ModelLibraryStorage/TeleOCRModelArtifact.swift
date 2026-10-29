import Foundation
import ModelLibraryFeature

public enum TeleOCRModelArtifact {
    public static let contract = "teleocr-dynamic-fp16-v1"
    public static let modelID = "XingChen-AGI/TeleOCR"
    public static let revision = "70a19ec6a3b036311af19ae5e762e650f41a5980"
    public static let manifestSHA256 = "a0783aadd9e824c525c6343352addc0765df7bd1857b9da6064b139cc025e56a"

    public static func make() throws -> ModelArtifact {
        guard let url = Bundle.module.url(forResource: "teleocr-fp16-files", withExtension: "json") else {
            throw ManagedModelStoreError.invalidDefinition
        }
        let files = try JSONDecoder().decode([ModelArtifactFile].self, from: Data(contentsOf: url))
        return ModelArtifact(
            id: "teleocr-fp16", title: "FP16", contract: contract, platforms: [.macOS],
            sizeBytes: files.reduce(0) { $0 + $1.bytes },
            upstream: ModelUpstream(repository: modelID, revision: revision), files: files,
            localBundle: ModelLocalBundle(
                manifestSHA256: manifestSHA256, profileID: "teleocr-fp16", manifestPath: "metadata.json"))
    }

    static func validateManifest(_ data: Data) throws {
        let metadata = try JSONDecoder().decode(Metadata.self, from: data)
        let artifact = try make()
        let expected = Dictionary(
            uniqueKeysWithValues: artifact.files.filter { $0.path != "metadata.json" }.map {
                ($0.path, $0.sha256)
            })
        guard metadata.format == "coreai-samples-teleocr/1", metadata.precision == "fp16",
            metadata.platform == nil || metadata.platform == "macos",
            metadata.source.modelID == modelID, metadata.source.revision == revision,
            metadata.assets == ["vision": "vision.aimodel", "embed": "embed.aimodel", "main": "main.aimodel"],
            metadata.files == expected
        else { throw ManagedModelStoreError.contractMismatch("TeleOCR metadata") }
    }

    private struct Metadata: Decodable {
        let format: String
        let precision: String
        let platform: String?
        let source: Source
        let assets: [String: String]
        let files: [String: String]
    }

    private struct Source: Decodable {
        let modelID: String
        let revision: String

        enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case revision
        }
    }
}
