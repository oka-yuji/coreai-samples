import Foundation
import ModelLibraryFeature
import Testing

struct TeleOCRNavigationTests {
    @Test("Mac OCR closing through background and sidebar transitions returns to the library after release")
    func closingDuringBackgroundStop() throws {
        let hash = String(repeating: "a", count: 64)
        let artifact = ModelArtifact(
            id: "teleocr-fp16", title: "FP16", contract: "teleocr-dynamic-fp16-v1", platforms: [.macOS],
            files: [ModelArtifactFile(path: "metadata.json", bytes: 0, sha256: hash)],
            localBundle: ModelLocalBundle(manifestSHA256: hash, profileID: "fixture", manifestPath: "metadata.json"))
        let catalog = try TeleOCRDemoCatalog.make(artifact: artifact)
        var state = ModelLibraryState(
            catalog: catalog, environment: LibraryEnvironment(platform: .macOS, connectedDemos: ["ocr"]), owner: UUID())
        _ = ModelLibraryFeature.reduce(&state, .refresh)
        let inspecting = try #require(state.installations[artifact.id]?.requestID)
        let resource = ValidatedModelResource(
            modelID: "teleocr", artifactID: artifact.id, contract: artifact.contract, revision: hash,
            upstreamRevision: nil, directory: URL(filePath: "/fixture/model"), sizeBytes: 0, sourceKind: .localManifest)
        _ = ModelLibraryFeature.reduce(&state, .inspected(artifact.id, inspecting, resource))
        _ = ModelLibraryFeature.reduce(&state, .openDemo(TeleOCRDemoCatalog.selection))
        let running = try #require(state.session)
        _ = ModelLibraryFeature.reduce(&state, .closeDemo)
        let stopping = try #require(state.stopRequestID)
        #expect(state.pendingDestination == .library)
        #expect(ModelLibraryFeature.reduce(&state, .closeDemo).isEmpty)
        #expect(state.stopRequestID == stopping)
        _ = ModelLibraryFeature.reduce(&state, .demoStopped(running.id, stopping))
        #expect(state.selection == nil && state.session == nil && state.pendingDestination == nil)
        #expect(!ModelLibraryFeature.reduce(&state, .deleteRequested(artifact.id)).isEmpty)
        #expect(state.failure == nil)
    }
}
