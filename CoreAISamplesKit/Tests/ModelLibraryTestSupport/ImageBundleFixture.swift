import CryptoKit
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import Testing

public struct ImageBundleFixture: Sendable {
    public let base: URL
    public let source: URL
    public let root: URL
    public let model: LibraryModel
    public let artifact: ModelArtifact
    public let files: [String: Data]

    public init(mutate: (inout [String: Any], inout [String: Data]) throws -> Void = { _, _ in }) throws {
        base = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        source = base.appending(path: "selected")
        root = base.appending(path: "managed")
        let url = try #require(
            Bundle.module.url(forResource: "image-profile-v2", withExtension: "json", subdirectory: "Resources"))
        let data = try Data(contentsOf: url)
        var profile = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        profile["id"] = "fixture-qmi-profile"
        let components = try #require(profile["components"] as? [String: [String: Any]])
        var contents: [String: Data] = [
            "LICENSE": Data("fixture license".utf8), "NOTICE.txt": Data("fixture attribution".utf8),
            "config/processor/tokenizer.json": Data("{}".utf8),
        ]
        for (kind, component) in components {
            let asset = try #require(component["asset"] as? String)
            contents[asset + "/weights.bin"] = Data(repeating: 7, count: 4096)
            let header: [String: Any] = [
                "format": 2, "kind": kind, "asset": URL(filePath: asset).lastPathComponent,
                "conversion_complete": true, "state_names": [], "profile": profile,
                "inputs": try #require(component["inputs"]), "outputs": try #require(component["outputs"]),
            ]
            contents[kind + "/component.json"] = try JSONSerialization.data(
                withJSONObject: header, options: .sortedKeys)
        }
        var manifest: [String: Any] = [
            "format": 2, "profile": profile, "verified": true,
            "numeric_verified": true, "quality_verified": true, "device_verification": "pending",
        ]
        try mutate(&manifest, &contents)
        manifest["assets"] = contents.mapValues { ["bytes": $0.count, "sha256": Self.hash($0)] as [String: Any] }
        let raw = try JSONSerialization.data(withJSONObject: manifest, options: .sortedKeys)
        contents["bundle.json"] = raw
        files = contents
        artifact = ModelArtifact(
            id: "qwen-fixture", title: "Image fixture", contract: "qwen-image21-mobile-v2", platforms: [.iOS],
            upstream: ModelUpstream(
                repository: "Qwen/Qwen-Image-2.1-Turbo", revision: "fc13ba1986fcb905854dc8de2f52eee395f49b5b"),
            files: contents.sorted { $0.key < $1.key }.map {
                ModelArtifactFile(path: $0.key, bytes: Int64($0.value.count), sha256: Self.hash($0.value))
            }, localBundle: ModelLocalBundle(manifestSHA256: Self.hash(raw), profileID: "fixture-qmi-profile"))
        model = LibraryModel(
            id: "qwen", name: "Qwen Image", category: "Image generation", artifacts: [artifact],
            defaultArtifactID: artifact.id, platforms: [.iOS], demos: [])
        _ = try ModelCatalog(models: [model], demos: [])
        for (name, bytes) in contents {
            let target = source.appending(path: name)
            try FileManager.default.createDirectory(
                at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bytes.write(to: target)
        }
    }

    public func store(available: Int64 = .max) -> ManagedModelStore {
        ManagedModelStore(root: root, availableCapacity: { _ in available })
    }

    public func install(into store: ManagedModelStore) async throws -> ValidatedModelResource {
        let staging = try await store.copyForImport(from: source, model: model, artifact: artifact, progress: { _ in })
        return try await store.promote(staging: staging, model: model, artifact: artifact)
    }

    public func remove() { try? FileManager.default.removeItem(at: base) }

    static func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
