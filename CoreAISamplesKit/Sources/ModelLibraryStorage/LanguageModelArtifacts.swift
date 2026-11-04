import CryptoKit
import Foundation
import ModelLibraryFeature

public enum LanguageModelArtifacts {
    public static let qwenID = "qwen3.8-27b-coreai-4bit"
    public static let clefID = "clef-flash-9b-coreai-decide"
    public static let qwenRevision = "8e24aebe036eae577d4a3bd019ba728a73358318"
    public static let clefRevision = "12e30be4e391310377c534524df8962359a08b0f"

    public static func qwen() throws -> ModelArtifact {
        try make(
            "qwen", id: qwenID, title: "4-bit", repository: "okayuji/Qwen3.8-27B-coreai-4bit",
            revision: qwenRevision, upstream: "Qwen/Qwen3.8-27B",
            upstreamRevision: "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    }

    public static func clef() throws -> ModelArtifact {
        try make(
            "clef", id: clefID, title: "uint4 + fp16 head", repository: "okayuji/clef-flash-9b-coreai-decide",
            revision: clefRevision, upstream: "Cloudflare/clef-flash",
            upstreamRevision: "17f0b0ad64efb65d273590632833508766b2aae6")
    }

    static func validateManifest(_ data: Data, decision: Bool) throws {
        let artifact = try decision ? clef() : qwen()
        let path = decision ? "decision.json" : "metadata.json"
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard artifact.files.first(where: { $0.path == path })?.sha256 == digest else {
            throw ManagedModelStoreError.contractMismatch(path)
        }
    }

    private static func make(
        _ name: String, id: String, title: String, repository: String, revision: String,
        upstream: String, upstreamRevision: String
    ) throws -> ModelArtifact {
        guard let url = Bundle.module.url(forResource: name + "-mac-files", withExtension: "json") else {
            throw ManagedModelStoreError.invalidDefinition
        }
        let files = try JSONDecoder().decode([ModelArtifactFile].self, from: Data(contentsOf: url))
        return ModelArtifact(
            id: id, title: title, contract: id + "-v1", platforms: [.macOS],
            sizeBytes: files.reduce(0) { $0 + $1.bytes },
            download: ModelDownload(repository: repository, revision: revision),
            upstream: ModelUpstream(repository: upstream, revision: upstreamRevision), files: files)
    }
}
