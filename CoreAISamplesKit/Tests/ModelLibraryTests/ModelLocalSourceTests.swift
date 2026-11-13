import Foundation
import ModelLibraryFeature
import Testing

struct ModelLocalSourceTests {
    private let hash = String(repeating: "a", count: 64)

    @Test("Legacy downloaded artifacts decode without a local source and retain their encoded fields")
    func legacyArtifactCompatibility() throws {
        let original = ModelArtifact(
            id: "old", title: "Old", contract: "legacy", platforms: [.iOS],
            download: ModelDownload(repository: "owner/model", revision: String(repeating: "b", count: 40)),
            files: [ModelArtifactFile(path: "weights.bin", bytes: 3, sha256: hash)])
        let bytes = try JSONEncoder().encode(original)
        let object = try #require(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        #expect(object["localBundle"] == nil)
        let restored = try JSONDecoder().decode(ModelArtifact.self, from: bytes)
        #expect(restored == original)
        #expect(restored.sourceKind == .huggingFace)
        #expect(restored.sourceRevision == original.download?.revision)
        #expect(restored.isSourceValid)
    }

    @Test("Local resources use the raw manifest hash and cannot masquerade as Hugging Face resources")
    func localSourceIdentity() throws {
        let artifact = localArtifact()
        let model = LibraryModel(
            id: "qwen-image", name: "Qwen Image", category: "Image generation", artifacts: [artifact],
            defaultArtifactID: artifact.id, platforms: [.iOS], demos: [])
        _ = try ModelCatalog(models: [model], demos: [])
        let directory = URL(filePath: "/fixture/model")
        let local = ValidatedModelResource(
            modelID: model.id, artifactID: artifact.id, contract: artifact.contract, revision: hash,
            upstreamRevision: nil, directory: directory, sizeBytes: 12, sourceKind: .localManifest)
        let downloaded = ValidatedModelResource(
            modelID: model.id, artifactID: artifact.id, contract: artifact.contract, revision: hash,
            upstreamRevision: nil, directory: directory, sizeBytes: 12)
        #expect(local.matches(model: model, artifact: artifact))
        #expect(!downloaded.matches(model: model, artifact: artifact))
        #expect(local.identity != downloaded.identity)
        #expect(artifact.download == nil)
        #expect(artifact.sourceRevision == hash)
    }

    @Test("Invalid locators, unknown manifest hashes and overflowing file sizes are rejected")
    func invalidSourceDefinitions() {
        let definitions = [
            localArtifact(
                download: ModelDownload(repository: "owner/model", revision: "main")),
            localArtifact(manifestHash: String(repeating: "b", count: 64)),
            localArtifact(manifestHash: String(repeating: "a", count: 40)),
            localArtifact(files: [
                ModelArtifactFile(path: "bundle.json", bytes: .max, sha256: hash),
                ModelArtifactFile(path: "weights.bin", bytes: 1, sha256: hash),
            ]),
        ]
        for artifact in definitions {
            let model = LibraryModel(
                id: "qwen-image", name: "Qwen Image", category: "Image generation", artifacts: [artifact],
                defaultArtifactID: artifact.id, platforms: [.iOS], demos: [])
            #expect(throws: ModelCatalogError.invalidArtifact(artifact.id)) {
                try ModelCatalog(models: [model], demos: [])
            }
        }
    }

    @Test(
        "CD-01 Every source combination preserves its trust identity and acquisition choices",
        arguments: [false, true], [false, true])
    func sourceCombinations(local: Bool, remote: Bool) throws {
        let locator = ModelDownload(
            repository: "owner/model", revision: String(repeating: "b", count: 40), root: "bundle")
        let artifact = ModelArtifact(
            id: "fixture", title: "Fixture", contract: "fixture", platforms: [.macOS],
            download: remote ? locator : nil,
            files: [ModelArtifactFile(path: "bundle.json", bytes: 12, sha256: hash)],
            localBundle: local ? ModelLocalBundle(manifestSHA256: hash, profileID: "fixture") : nil)
        _ = try ModelCatalog(models: [model(artifact)], demos: [])
        #expect(artifact.isSourceValid == (local || remote))
        #expect(artifact.sourceKind == (local ? .localManifest : remote ? .huggingFace : nil))
        #expect(artifact.sourceRevision == (local ? hash : remote ? locator.revision : nil))
        #expect(try JSONDecoder().decode(ModelArtifact.self, from: JSONEncoder().encode(artifact)) == artifact)
        let policy = ModelAcquisitionPolicy(artifact: artifact)
        #expect(policy.preferredMethod == (remote ? .download : local ? .importBundle : nil))
        #expect(policy.allowsImport == local)
        if local && remote { #expect(policy.readyTitle == "Model ready") }
        if local && !remote { #expect(policy.readyTitle == "Imported") }
        if remote && !local { #expect(policy.readyTitle == "Downloaded") }
    }

    @Test(
        "CD-01 Invalid download locators cannot be hidden by a valid manifest",
        arguments: [
            ("owner/model/extra", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", ""),
            ("../model", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", ""),
            ("owner/model", "main", ""),
            ("owner/model", String(repeating: "b", count: 64), ""),
            ("owner/model", String(repeating: "B", count: 40), ""),
            ("owner/model", String(repeating: "b", count: 40), "../bundle"),
            ("owner/model", String(repeating: "b", count: 40), "/bundle"),
            ("owner/model", String(repeating: "b", count: 40), "bundle//v1"),
        ])
    func invalidLocator(repository: String, revision: String, root: String) {
        let locator = ModelDownload(repository: repository, revision: revision, root: root)
        let artifact = localArtifact(download: locator)
        #expect(!locator.isValid && !artifact.isSourceValid)
        #expect(ModelAcquisitionPolicy(artifact: artifact).preferredMethod == nil)
        #expect(!ModelAcquisitionPolicy(artifact: artifact).allowsImport)
        #expect(throws: ModelCatalogError.invalidArtifact(artifact.id)) {
            try ModelCatalog(models: [model(artifact)], demos: [])
        }
    }

    @Test(
        "CD-01 Downloadable manifest artifacts reject invalid inventories",
        arguments: ["missing", "hash", "duplicate", "path", "negative", "overflow"])
    func hybridInventory(_ defect: String) {
        var files = [ModelArtifactFile(path: "bundle.json", bytes: 12, sha256: hash)]
        switch defect {
        case "missing": files = []
        case "hash":
            files = [ModelArtifactFile(path: "bundle.json", bytes: 12, sha256: String(repeating: "c", count: 64))]
        case "duplicate": files += files
        case "path": files.append(ModelArtifactFile(path: "../weights", bytes: 1, sha256: hash))
        case "negative": files.append(ModelArtifactFile(path: "weights", bytes: -1, sha256: hash))
        default: files.append(ModelArtifactFile(path: "weights", bytes: .max, sha256: hash))
        }
        let artifact = localArtifact(
            download: ModelDownload(repository: "owner/model", revision: String(repeating: "b", count: 40)),
            files: files)
        #expect(throws: ModelCatalogError.invalidArtifact(artifact.id)) {
            try ModelCatalog(models: [model(artifact)], demos: [])
        }
    }

    @Test("CD-02 Adding a locator preserves an imported resource and never downloads an already ready artifact")
    func importedCompatibility() throws {
        let old = localArtifact()
        let hybrid = localArtifact(
            download: ModelDownload(repository: "owner/model", revision: String(repeating: "b", count: 40)))
        let resource = ValidatedModelResource(
            modelID: "fixture", artifactID: old.id, contract: old.contract, revision: hash, upstreamRevision: nil,
            directory: URL(filePath: "/managed/unchanged"), sizeBytes: 12, sourceKind: .localManifest)
        #expect(old.files == hybrid.files && old.localBundle == hybrid.localBundle)
        #expect(resource.matches(model: model(old), artifact: old))
        #expect(resource.matches(model: model(hybrid), artifact: hybrid))
        var state = ModelLibraryState(
            catalog: try acquisitionCatalog(hybrid),
            environment: LibraryEnvironment(platform: .iOS), owner: UUID())
        _ = ModelLibraryFeature.reduce(&state, .refresh)
        let ticket = try #require(state.installations[hybrid.id]?.requestID)
        #expect(ModelLibraryFeature.reduce(&state, .inspected(hybrid.id, ticket, resource)).isEmpty)
        let ready = state
        #expect(ModelLibraryFeature.reduce(&state, .downloadRequested(hybrid.id)).isEmpty)
        #expect(state == ready && state.installations[hybrid.id]?.resource == resource && state.session == nil)
    }

    @Test("CD-03 Progress and cancellation labels follow the current transfer phase")
    func transferPresentation() {
        let downloading = ModelAcquisitionPolicy.transferMethod(for: .downloading(nil))
        let importing = ModelAcquisitionPolicy.transferMethod(for: .importing(nil))
        #expect(downloading?.progressTitle == "Downloading" && downloading?.cancellationTitle == "Cancel download")
        #expect(importing?.progressTitle == "Importing" && importing?.cancellationTitle == "Cancel import")
        #expect(downloading?.actionTitle(retrying: true) == "Retry download")
        #expect(importing?.actionTitle(retrying: true) == "Import model")
        for phase: ModelInstallationPhase in [
            .missing, .ready, .failed, .validating, .inspecting, .stopping, .deleting,
        ] {
            #expect(ModelAcquisitionPolicy.transferMethod(for: phase) == nil)
        }
    }

    @Test(
        "CD-03 Acquisition keeps model, artifact and platform visibility restrictions",
        arguments: ["model", "artifact", "platform"])
    func acquisitionVisibility(_ restriction: String) throws {
        let visible = localArtifact(
            download: ModelDownload(repository: "owner/model", revision: String(repeating: "b", count: 40)))
        let artifact = ModelArtifact(
            id: visible.id, title: visible.title, contract: visible.contract, platforms: visible.platforms,
            download: visible.download, files: visible.files,
            visibility: restriction == "artifact" ? .hidden : .published, localBundle: visible.localBundle)
        var state = ModelLibraryState(
            catalog: try acquisitionCatalog(
                artifact, visibility: restriction == "model" ? .hidden : .published, platforms: [.iOS]),
            environment: LibraryEnvironment(platform: restriction == "platform" ? .macOS : .iOS), owner: UUID())
        #expect(ModelLibraryFeature.reduce(&state, .refresh).isEmpty)
        #expect(ModelLibraryFeature.reduce(&state, .downloadRequested(artifact.id)).isEmpty)
        #expect(
            ModelLibraryFeature.reduce(&state, .importRequested(artifact.id, URL(filePath: "/selected/model"))).isEmpty)
        #expect(state.installations[artifact.id] == ModelInstallation() && state.session == nil)
    }

    @Test("CD-03 Models without visible demo links remain outside library acquisition")
    func modelsWithoutDemos() throws {
        let artifact = localArtifact(
            download: ModelDownload(repository: "owner/model", revision: String(repeating: "b", count: 40)))
        var state = ModelLibraryState(
            catalog: try ModelCatalog(models: [model(artifact)], demos: []),
            environment: LibraryEnvironment(platform: .iOS), owner: UUID())
        #expect(ModelLibraryFeature.reduce(&state, .refresh).isEmpty)
        #expect(ModelLibraryFeature.reduce(&state, .downloadRequested(artifact.id)).isEmpty)
        #expect(
            ModelLibraryFeature.reduce(&state, .importRequested(artifact.id, URL(filePath: "/selected/model"))).isEmpty)
        #expect(state.installations[artifact.id] == ModelInstallation())
    }

    private func acquisitionCatalog(
        _ artifact: ModelArtifact, visibility: CatalogVisibility = .published,
        platforms: Set<LibraryPlatform> = [.iOS, .macOS]
    ) throws -> ModelCatalog {
        let demo = DemoDescriptor(
            id: "forecast", title: "Forecast", systemImage: "chart.line.uptrend.xyaxis", summary: "Forecast",
            platforms: platforms)
        let model = LibraryModel(
            id: "fixture", name: "Fixture", category: "Forecasting", artifacts: [artifact],
            defaultArtifactID: artifact.id, platforms: platforms, visibility: visibility,
            demos: [ModelDemoLink(demoID: demo.id, platforms: platforms)])
        return try ModelCatalog(models: [model], demos: [demo])
    }

    private func model(_ artifact: ModelArtifact) -> LibraryModel {
        LibraryModel(
            id: "fixture", name: "Fixture", category: "Forecasting", artifacts: [artifact],
            defaultArtifactID: artifact.id, platforms: [.iOS, .macOS], demos: [])
    }

    private func localArtifact(
        download: ModelDownload? = nil, manifestHash: String? = nil, files: [ModelArtifactFile]? = nil
    ) -> ModelArtifact {
        ModelArtifact(
            id: "qwen-mobile", title: "Mobile", contract: "qwen-image21-mobile-v2", platforms: [.iOS],
            download: download, files: files ?? [ModelArtifactFile(path: "bundle.json", bytes: 12, sha256: hash)],
            localBundle: ModelLocalBundle(manifestSHA256: manifestHash ?? hash, profileID: "qmi-mobile-w4-v1"))
    }
}
