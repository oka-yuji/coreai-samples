import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryRuntime
import ModelLibraryStorage
import Testing

@MainActor
struct LocalSpeechCatalogTests {
    @Test("IOS-IR-06 A catalog construction failure writes a diagnostic report and returns failure")
    func catalogFailure() async throws {
        let root = URL.temporaryDirectory.appending(path: "speech-catalog-error-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appending(path: "report.json")
        let status = await ModelLibrarySelfTest.run(output: output, root: root) {
            throw CatalogFixtureFailure.missingAssets
        }
        #expect(status == 1)
        let report = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: output)) as? [String: Any])
        let failures = try #require(report["failures"] as? [String])
        #expect(failures.contains { $0.contains("missingAssets") })
    }

    @Test(
        "IR-SESSION-01 The speech catalog uses the pinned local artifact and the diagnostic checks the supplied catalog"
    )
    func catalog() async throws {
        let artifact = try IrodoriModelArtifact.make()
        let catalog = try IOSDemoCatalog.make(speechArtifact: artifact)
        let model = try #require(catalog.model("irodori"))
        #expect(model.artifacts == [artifact])
        #expect(model.demos.map(\.demoID) == ["speech"])
        #expect(model.demos.allSatisfy { $0.requiredCapabilities.isEmpty })
        let registry = ModelDemoAdapterRegistry()
        let labels = OwnerLabels()
        let (_, sink) = AsyncStream.makeStream(of: String.self)
        for demo in catalog.demos { registry.register(FakeAdapter(demo.id, label: labels, sink: sink)) }
        try catalog.validateConnections(adapterIDs: Array(registry.connectedDemos))
        let root = URL.temporaryDirectory.appending(path: "speech-catalog-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appending(path: "report.json")
        let status = await ModelLibrarySelfTest.run(
            output: output, root: root.appending(path: "managed"), registry: registry, capabilities: [:],
            catalog: { catalog })
        #expect(status == 0)
        let report = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: output)) as? [String: Any])
        #expect(report["visibleModels"] as? Int == 3)
        #expect(report["visibleDemos"] as? Int == 7)
        let resources = try #require(report["artifacts"] as? [[String: Any]])
        let speech = try #require(resources.first { $0["modelID"] as? String == "irodori" })
        #expect(speech["sourceKind"] as? String == "localManifest")
        #expect(speech["localManifestSHA256"] as? String == artifact.sourceRevision)
        #expect(speech["files"] as? Int == 14)
    }
}

private enum CatalogFixtureFailure: Error { case missingAssets }
