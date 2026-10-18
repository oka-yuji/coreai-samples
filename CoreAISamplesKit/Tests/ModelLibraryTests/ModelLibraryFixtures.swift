import Foundation
import ModelLibraryFeature
import Testing

struct LibraryFixture {
    static let distributionRevision = "0123456789abcdef0123456789abcdef01234567"
    static let upstreamRevision = "89abcdef0123456789abcdef0123456789abcdef"
    static let checksum = String(repeating: "c", count: 64)

    let owner = UUID()

    func artifact(
        _ id: String, visibility: CatalogVisibility = .published, platforms: Set<LibraryPlatform> = [.iOS],
        root: String = "", files: [ModelArtifactFile]? = nil
    ) -> ModelArtifact {
        ModelArtifact(
            id: id, title: id.uppercased(), contract: "fixture-" + id, platforms: platforms, sizeBytes: 4,
            download: ModelDownload(repository: "owner/fixture", revision: Self.distributionRevision, root: root),
            upstream: ModelUpstream(repository: "upstream/fixture", revision: Self.upstreamRevision),
            files: files ?? [ModelArtifactFile(path: "bundle/model.bin", bytes: 4, sha256: Self.checksum)],
            visibility: visibility)
    }

    func demo(_ id: String) -> DemoDescriptor {
        DemoDescriptor(
            id: id, title: id + " demo", systemImage: "doc.text", summary: "Fixture " + id,
            explanation: [DemoExplanation(title: "How it works", text: "Fixture explanation for " + id)],
            platforms: [.iOS])
    }

    func model(_ id: String, artifacts: [ModelArtifact], defaultArtifactID: String) -> LibraryModel {
        LibraryModel(
            id: id, name: id, category: "Fixture", artifacts: artifacts, defaultArtifactID: defaultArtifactID,
            platforms: [.iOS], demos: [ModelDemoLink(demoID: "text", platforms: [.iOS])])
    }

    func variantCatalog() throws -> ModelCatalog {
        let model = LibraryModel(
            id: "model", name: "Fixture model", category: "Fixture",
            artifacts: [
                artifact("fp32"), artifact("int4", visibility: .development), artifact("mac", platforms: [.macOS]),
            ],
            defaultArtifactID: "fp32", platforms: [.iOS],
            demos: [
                ModelDemoLink(
                    demoID: "camera", platforms: [.iOS], requiredCapabilities: [.camera],
                    optionalCapabilities: [.microphone]),
                ModelDemoLink(demoID: "text", platforms: [.iOS]),
            ])
        return try ModelCatalog(models: [model], demos: [demo("camera"), demo("text")])
    }

    func registeredCatalog(_ entries: [(model: String, demos: [String])]) throws -> ModelCatalog {
        var demoIDs: [String] = []
        for id in entries.flatMap(\.demos) where !demoIDs.contains(id) { demoIDs.append(id) }
        let models = entries.map { entry in
            LibraryModel(
                id: entry.model, name: entry.model, category: "Fixture", artifacts: [artifact(entry.model + "-fp32")],
                defaultArtifactID: entry.model + "-fp32", platforms: [.iOS],
                demos: entry.demos.map { ModelDemoLink(demoID: $0, platforms: [.iOS]) })
        }
        return try ModelCatalog(models: models, demos: demoIDs.map(demo))
    }

    func state(
        _ catalog: ModelCatalog, platform: LibraryPlatform = .iOS, audience: LibraryAudience = .release,
        connected: Set<String>? = nil, capabilities: [DemoCapability: CapabilityState] = [.camera: .available]
    ) -> ModelLibraryState {
        let environment = LibraryEnvironment(
            platform: platform, audience: audience, connectedDemos: connected ?? Set(catalog.demos.map(\.id)),
            capabilities: capabilities)
        return ModelLibraryState(catalog: catalog, environment: environment, owner: owner)
    }

    func resource(
        _ artifactID: String, modelID: String = "model", revision: String = distributionRevision,
        upstream: String? = upstreamRevision
    ) -> ValidatedModelResource {
        ValidatedModelResource(
            modelID: modelID, artifactID: artifactID, contract: "fixture-" + artifactID, revision: revision,
            upstreamRevision: upstream, directory: URL(filePath: "/fixture/" + artifactID), sizeBytes: 4)
    }

    func install(_ state: inout ModelLibraryState, _ artifactID: String, modelID: String = "model") throws {
        #expect(ModelLibraryFeature.reduce(&state, .downloadRequested(artifactID)).count == 1)
        let request = try #require(state.installations[artifactID]?.requestID)
        let directory = URL(filePath: "/fixture/" + artifactID)
        #expect(ModelLibraryFeature.reduce(&state, .downloaded(artifactID, request, directory)).count == 1)
        let validated = ModelLibraryFeature.reduce(
            &state, .validated(artifactID, request, resource(artifactID, modelID: modelID)))
        #expect(validated.isEmpty)
        #expect(state.installations[artifactID]?.phase == .ready)
    }
}
