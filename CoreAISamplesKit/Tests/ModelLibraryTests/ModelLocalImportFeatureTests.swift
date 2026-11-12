import Foundation
import ModelLibraryFeature
import Testing

struct ModelLocalImportFeatureTests {
    private let source = URL(filePath: "/selected/bundle")

    @Test(
        "QMI-06 A local bundle reaches ready without creating a session for an unconnected demo",
        arguments: [false, true])
    func localImportWithoutAdapter(downloadable: Bool) throws {
        var fixture = try fixture(downloadable: downloadable)
        let (model, artifact) = (fixture.model, fixture.artifact)
        #expect(fixture.state.variantTitle(for: model.id) == artifact.title)
        let effects = ModelLibraryFeature.reduce(&fixture.state, .importRequested(artifact.id, source))
        let ticket = try #require(fixture.state.installations[artifact.id]?.requestID)
        #expect(effects == [.run(.importBundle(model, artifact, ticket, source), id: .resource(artifact.id))])
        #expect(fixture.state.installations[artifact.id]?.phase == .importing(nil))
        let complete = ModelLibraryFeature.reduce(&fixture.state, .importCopied(artifact.id, ticket))
        #expect(complete.isEmpty)
        #expect(fixture.state.installations[artifact.id]?.phase == .validating)
        #expect(ModelLibraryFeature.reduce(&fixture.state, .validated(artifact.id, ticket, fixture.resource)).isEmpty)
        #expect(fixture.state.installations[artifact.id]?.phase == .ready)
        let selection = DemoSelection(modelID: model.id, demoID: "image-generation")
        _ = ModelLibraryFeature.reduce(&fixture.state, .openDemo(selection))
        #expect(fixture.state.session == nil)
        #expect(fixture.state.access(to: selection) == .preview(.ready))
    }

    @Test(
        "QMI-06 Missing adapters are allowed only for explicit previews while duplicate and unknown adapters are rejected"
    )
    func declaredConnections() throws {
        let catalog = try fixture().state.catalog
        try catalog.validateConnections(adapterIDs: [], unconnectedDemoIDs: ["image-generation"])
        try catalog.validateConnections(adapterIDs: ["image-generation"])
        #expect(throws: ModelCatalogError.invalidDemo("image-generation")) {
            try catalog.validateConnections(adapterIDs: [])
        }
        #expect(throws: ModelCatalogError.invalidDemo("unknown")) {
            try catalog.validateConnections(adapterIDs: ["unknown"])
        }
        #expect(throws: ModelCatalogError.duplicateDemo("image-generation")) {
            try catalog.validateConnections(adapterIDs: ["image-generation", "image-generation"])
        }
        #expect(throws: ModelCatalogError.duplicateDemo("image-generation")) {
            try catalog.validateConnections(adapterIDs: ["image-generation"], unconnectedDemoIDs: ["image-generation"])
        }
    }

    @Test(
        "QMI-03 Cancelling a replacement waits for inspection and rejects late copy and validation results",
        arguments: [false, true])
    func cancelKeepsVerifiedPriorResource(downloadable: Bool) throws {
        var fixture = try fixture(installed: true, downloadable: downloadable)
        let id = fixture.artifact.id
        _ = ModelLibraryFeature.reduce(&fixture.state, .importRequested(id, source))
        let ticket = try #require(fixture.state.installations[id]?.requestID)
        #expect(
            ModelLibraryFeature.reduce(&fixture.state, .cancelRequested(id)) == [
                .cancel(.resource(id)), .run(.stopImport(fixture.model, fixture.artifact, ticket), id: .stopping(id)),
            ])
        #expect(fixture.state.installations[id]?.phase == .stopping)
        #expect(ModelLibraryFeature.reduce(&fixture.state, .importCopied(id, ticket)).isEmpty)
        #expect(ModelLibraryFeature.reduce(&fixture.state, .transferStopped(id, ticket)).isEmpty)
        #expect(fixture.state.installations[id]?.phase == .stopping)
        _ = ModelLibraryFeature.reduce(&fixture.state, .importStopped(id, ticket, fixture.resource))
        #expect(fixture.state.installations[id]?.phase == .ready)
        #expect(fixture.state.installations[id]?.resource == fixture.resource)
        #expect(ModelLibraryFeature.reduce(&fixture.state, .validated(id, ticket, fixture.resource)).isEmpty)
    }

    @Test(
        "QMI-03 Failed replacement rechecks the previous installation instead of trusting a retained resource",
        arguments: [false, true])
    func failureRechecksStorage(downloadable: Bool) throws {
        for restored in [true, false] {
            var fixture = try fixture(installed: true, downloadable: downloadable)
            let id = fixture.artifact.id
            _ = ModelLibraryFeature.reduce(&fixture.state, .importRequested(id, source))
            let ticket = try #require(fixture.state.installations[id]?.requestID)
            let effects = ModelLibraryFeature.reduce(&fixture.state, .failed(id, ticket, .invalidResource))
            let inspection = try #require(fixture.state.installations[id]?.requestID)
            #expect(inspection != ticket)
            #expect(effects == [.run(.inspect(fixture.model, fixture.artifact, inspection), id: .resource(id))])
            #expect(fixture.state.installations[id]?.phase == .inspecting)
            #expect(ModelLibraryFeature.reduce(&fixture.state, .importCopied(id, ticket)).isEmpty)
            _ = ModelLibraryFeature.reduce(
                &fixture.state, .inspected(id, inspection, restored ? fixture.resource : nil))
            #expect(fixture.state.installations[id]?.phase == (restored ? .ready : .failed))
            #expect(fixture.state.installations[id]?.resource == (restored ? fixture.resource : nil))
            #expect(fixture.state.installations[id]?.failure == .invalidResource)
        }
    }

    @Test(
        "QMI-04 A current demo prevents replacement and a resource from the wrong source never becomes ready",
        arguments: [false, true])
    func liveDemoAndWrongSource(downloadable: Bool) throws {
        var fixture = try fixture(installed: true, connected: true, downloadable: downloadable)
        let id = fixture.artifact.id
        let selection = DemoSelection(modelID: fixture.model.id, demoID: "image-generation")
        _ = ModelLibraryFeature.reduce(&fixture.state, .openDemo(selection))
        #expect(fixture.state.session != nil)
        #expect(ModelLibraryFeature.reduce(&fixture.state, .importRequested(id, source)).isEmpty)
        #expect(fixture.state.installations[id]?.failure == .modelInUse)
        var other = try self.fixture(downloadable: downloadable)
        _ = ModelLibraryFeature.reduce(&other.state, .importRequested(id, source))
        let ticket = try #require(other.state.installations[id]?.requestID)
        _ = ModelLibraryFeature.reduce(&other.state, .importCopied(id, ticket))
        let incorrect = ValidatedModelResource(
            modelID: other.model.id, artifactID: id, contract: other.artifact.contract,
            revision: other.resource.revision, upstreamRevision: nil, directory: other.resource.directory, sizeBytes: 12
        )
        _ = ModelLibraryFeature.reduce(&other.state, .validated(id, ticket, incorrect))
        #expect(other.state.installations[id]?.phase == .failed)
        #expect(other.state.installations[id]?.resource == nil)
    }

    @Test("CD-02 A download after a cancelled import uses transfer cancellation and rejects the previous completion")
    func switchesAcquisitionRoute() throws {
        var fixture = try fixture(downloadable: true)
        let id = fixture.artifact.id
        _ = ModelLibraryFeature.reduce(&fixture.state, .importRequested(id, source))
        let importing = try #require(fixture.state.installations[id]?.requestID)
        _ = ModelLibraryFeature.reduce(&fixture.state, .cancelRequested(id))
        _ = ModelLibraryFeature.reduce(&fixture.state, .importStopped(id, importing, nil))
        let operations = ModelLibraryFeature.reduce(&fixture.state, .downloadRequested(id))
        let downloading = try #require(fixture.state.installations[id]?.requestID)
        #expect(downloading != importing && fixture.state.installations[id]?.phase == .downloading(nil))
        #expect(operations == [.run(.download(fixture.model, fixture.artifact, downloading), id: .resource(id))])
        _ = ModelLibraryFeature.reduce(&fixture.state, .downloaded(id, downloading, URL(filePath: "/staging/model")))
        #expect(fixture.state.installations[id]?.phase == .validating)
        #expect(
            ModelLibraryFeature.reduce(&fixture.state, .cancelRequested(id)) == [
                .cancel(.resource(id)), .run(.stopTransfer(artifactID: id, downloading), id: .stopping(id)),
            ])
        let stopping = fixture.state
        _ = ModelLibraryFeature.reduce(&fixture.state, .importStopped(id, importing, fixture.resource))
        _ = ModelLibraryFeature.reduce(&fixture.state, .validated(id, importing, fixture.resource))
        #expect(fixture.state == stopping)
        _ = ModelLibraryFeature.reduce(&fixture.state, .transferStopped(id, downloading))
        #expect(fixture.state.installations[id] == ModelInstallation() && fixture.state.session == nil)
    }

    private func fixture(installed: Bool = false, connected: Bool = false, downloadable: Bool = false) throws
        -> LocalFeatureFixture
    {
        let hash = String(repeating: "a", count: 64)
        let artifact = ModelArtifact(
            id: "qwen-mobile", title: "Mobile", contract: "qwen-image21-mobile-v2", platforms: [.iOS],
            download: downloadable
                ? ModelDownload(repository: "owner/model", revision: String(repeating: "b", count: 40)) : nil,
            files: [ModelArtifactFile(path: "bundle.json", bytes: 12, sha256: hash)],
            localBundle: ModelLocalBundle(manifestSHA256: hash, profileID: "fixture-qmi-profile"))
        let model = LibraryModel(
            id: "qwen", name: "Qwen Image", category: "Image generation", artifacts: [artifact],
            defaultArtifactID: artifact.id, platforms: [.iOS],
            demos: [ModelDemoLink(demoID: "image-generation", platforms: [.iOS])])
        let demo = DemoDescriptor(
            id: "image-generation", title: "Image generation", systemImage: "photo",
            summary: "Generate images", platforms: [.iOS])
        let catalog = try ModelCatalog(models: [model], demos: [demo])
        var state = ModelLibraryState(
            catalog: catalog,
            environment: LibraryEnvironment(platform: .iOS, connectedDemos: connected ? [demo.id] : []), owner: UUID())
        let resource = ValidatedModelResource(
            modelID: model.id, artifactID: artifact.id, contract: artifact.contract, revision: hash,
            upstreamRevision: nil, directory: URL(filePath: "/managed/artifact"), sizeBytes: 12,
            sourceKind: .localManifest)
        if installed {
            _ = ModelLibraryFeature.reduce(&state, .refresh)
            let ticket = try #require(state.installations[artifact.id]?.requestID)
            _ = ModelLibraryFeature.reduce(&state, .inspected(artifact.id, ticket, resource))
        }
        return LocalFeatureFixture(state: state, model: model, artifact: artifact, resource: resource)
    }
}

private struct LocalFeatureFixture {
    var state: ModelLibraryState
    let model: LibraryModel
    let artifact: ModelArtifact
    let resource: ValidatedModelResource
}
