import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryRuntime
import ModelLibraryStorage
import SpeechRuntime
import Testing

@MainActor
struct LocalSpeechCatalogTests {
    @Test(
        "IRDL-03 The public reference release downloads through Models and becomes ready after verification",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_IRODORI_DOWNLOAD_OUTPUT"] != nil))
    func downloadReference() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["COREAI_IRODORI_DOWNLOAD_OUTPUT"])
        let output = URL(filePath: path, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        let artifact = try IrodoriModelArtifact.makeReferenceDownload()
        let catalog = try IOSDemoCatalog.make(referenceSpeechArtifact: artifact)
        let model = try #require(catalog.model("irodori"))
        let root = output.appending(path: "ModelLibrary")
        let status = await ModelLibrarySelfTest.run(
            output: output.appending(path: "acquisition.json"), root: root, acquire: artifact.id,
            catalog: { catalog })
        #expect(status == 0)
        let result = try await ManagedModelStore(root: root).inspect(model: model, artifact: artifact)
        guard case .ready(let resource) = result else {
            Issue.record("The downloaded model did not pass inspection.")
            return
        }
        #expect(resource.revision == artifact.sourceRevision)
        #expect(resource.sizeBytes == artifact.totalFileBytes)
    }

    @Test("REF-UI-04 Models selects the reference variant and enables voice input without requiring microphone access")
    func referenceVariant() async throws {
        let legacy = try IrodoriModelArtifact.make()
        let reference = try IrodoriModelArtifact.make(profile: .referenceVoice)
        let catalog = try IOSDemoCatalog.make(speechArtifact: legacy, referenceSpeechArtifact: reference)
        let state = ModelLibraryState(catalog: catalog, environment: LibraryEnvironment(platform: .iOS), owner: UUID())
        #expect(state.selectedArtifact(for: "irodori") == reference)
        let model = try #require(catalog.model("irodori"))
        #expect(model.artifacts == [reference, legacy])
        #expect(model.demos.allSatisfy { $0.requiredCapabilities.isEmpty })
        let resource = ValidatedModelResource(
            modelID: model.id, artifactID: reference.id, contract: reference.contract,
            revision: try #require(reference.sourceRevision), upstreamRevision: nil,
            directory: URL(filePath: "/reference-fixture"), sizeBytes: reference.sizeBytes ?? 0,
            sourceKind: .localManifest)
        let session = ModelDemoSession(
            id: LibraryTicket(owner: UUID(), number: 1),
            selection: DemoSelection(modelID: "irodori", demoID: "speech"), resource: resource)
        let sessions = SpeechDemoSessions()
        let store = sessions.bind(session, examples: [], permits: { true })
        #expect(store.state.model?.supportsReferenceAudio == true && store.state.canSelectReference)
        await sessions.stop(session)
    }

    @Test("GUIDE-01 Model preparation and speech instructions remain available before importing a model")
    func firstUseGuidance() throws {
        let artifact = try IrodoriModelArtifact.make()
        let catalog = try IOSDemoCatalog.make(speechArtifact: artifact)
        var state = ModelLibraryState(catalog: catalog, environment: LibraryEnvironment(platform: .iOS), owner: UUID())
        let selection = DemoSelection(modelID: "irodori", demoID: "speech")
        #expect(ModelLibraryFeature.reduce(&state, .openDemo(selection)).isEmpty)
        #expect(state.access(to: selection) == .preview(.missing))
        #expect(state.session == nil)
        let explanation = try #require(state.catalog.demo("speech")).explanation
        #expect(explanation.count >= 4)
        #expect(explanation.contains { $0.text.contains("Import model") && $0.text.contains("3.38 GB") })
        #expect(explanation.contains { $0.text.contains("Try an example") })
        #expect(explanation.contains { $0.text.contains("Play") && $0.text.contains("Save WAV") })
        #expect(explanation.contains { $0.text.contains("reference voice") && $0.text.contains("not supported") })
    }

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
        #expect(report["visibleModels"] as? Int == 4)
        #expect(report["visibleDemos"] as? Int == 8)
        let resources = try #require(report["artifacts"] as? [[String: Any]])
        let speech = try #require(resources.first { $0["modelID"] as? String == "irodori" })
        #expect(speech["sourceKind"] as? String == "localManifest")
        #expect(speech["localManifestSHA256"] as? String == artifact.sourceRevision)
        #expect(speech["files"] as? Int == 14)
    }
}

private enum CatalogFixtureFailure: Error { case missingAssets }
