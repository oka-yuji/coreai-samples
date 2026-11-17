import Foundation
import ModelLibraryFeature
import SpeechAssets
import Testing

@testable import ModelLibraryStorage

struct VoxCPMDownloadTests {
    @Test("VDL-01 The Mac catalog downloads VoxCPM2 from the pinned fp16 folder with the bundled inventory")
    func catalogDownload() async throws {
        let catalog = try MacDemoCatalog.make()
        let model = try #require(catalog.model("voxcpm2"))
        let artifact = try #require(model.artifacts.first)
        let local = try VoxCPMModelArtifact.make()
        let inventory = try VoxCPMBundleInventory.files(profile: VoxCPMModelArtifact.profile)
        #expect(model.artifacts == [try VoxCPMModelArtifact.makeDownload()])
        #expect(
            artifact.download
                == ModelDownload(
                    repository: "okayuji/voxcpm2-coreai", revision: "afe11d61de9dd7c3609140aa040b777ef77b2555",
                    root: "fp16"))
        #expect(
            artifact.files == inventory.map { ModelArtifactFile(path: $0.path, bytes: $0.bytes, sha256: $0.sha256) })
        #expect(artifact.files.count == 22 && artifact.totalFileBytes == 4_677_413_664)
        #expect(artifact.sizeBytes == artifact.totalFileBytes)
        #expect(artifact.id == local.id && artifact.title == local.title && artifact.contract == local.contract)
        #expect(artifact.platforms == [.macOS] && artifact.upstream == local.upstream)
        #expect(artifact.localBundle == local.localBundle && artifact.sourceKind == .localManifest)
        #expect(artifact.sourceRevision == local.sourceRevision && artifact.isSourceValid)
        let policy = ModelAcquisitionPolicy(artifact: artifact)
        #expect(policy.preferredMethod == .download && !policy.allowsImport && policy.readyTitle == "Model ready")
        var state = ModelLibraryState(
            catalog: catalog, environment: LibraryEnvironment(platform: .macOS), owner: UUID())
        let effects = ModelLibraryFeature.reduce(&state, .downloadRequested(artifact.id))
        let request = try #require(state.installations[artifact.id]?.requestID)
        #expect(effects == [.run(.download(model, artifact, request), id: .resource(artifact.id))])
        let files = StoreFixture()
        defer { files.remove() }
        let store = files.store()
        #expect(try await store.inspect(model: model, artifact: artifact).isMissing)
        #expect(
            try await store.stagingDirectory(for: model, artifact: artifact).lastPathComponent
                == local.sourceRevision)
    }

    @Test("VDL-01 Only the published fp16 configuration has a download folder")
    func downloadFolders() {
        #expect(VoxCPMModelArtifact.downloadFolder(for: .fp16) == "fp16")
        #expect(VoxCPMModelArtifact.downloadFolder(for: .fp32) == nil)
        #expect(VoxCPMModelArtifact.downloadFolder(for: .fp16Proj32) == nil)
        #expect(VoxCPMModelArtifact.downloadFolder(for: VoxCPMModelArtifact.profile) != nil)
    }

    @Test(
        "VDL-02 A download definition that differs from the pinned inventory is never trusted",
        arguments: ["repository", "revision", "root", "hash", "bytes", "missing", "extra", "local"])
    func forgedDownload(_ field: String) async throws {
        let known = try VoxCPMModelArtifact.makeDownload()
        let source = try #require(known.download)
        var files = known.files
        switch field {
        case "hash":
            files[0] = ModelArtifactFile(path: files[0].path, bytes: files[0].bytes, sha256: sha256(Data("x".utf8)))
        case "bytes":
            files[0] = ModelArtifactFile(path: files[0].path, bytes: files[0].bytes + 1, sha256: files[0].sha256)
        case "missing": files.removeAll { $0.path == VoxCPMBundleInventory.tokenizerPath }
        case "extra": files.append(ModelArtifactFile(path: "extra.bin", bytes: 0, sha256: sha256(Data())))
        default: break
        }
        let artifact = ModelArtifact(
            id: known.id, title: known.title, contract: known.contract, platforms: known.platforms,
            sizeBytes: files.reduce(0) { $0 + $1.bytes },
            download: ModelDownload(
                repository: field == "repository" ? "other/voxcpm2-coreai" : source.repository,
                revision: field == "revision" ? String(repeating: "a", count: 40) : source.revision,
                root: field == "root" ? "fp32" : source.root),
            upstream: known.upstream, files: files, localBundle: field == "local" ? nil : known.localBundle)
        #expect(artifact.isSourceValid && artifact != known)
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let model = VoxCPMImportFixture.model(artifact)
        let rejection = ManagedModelStoreError.contractMismatch("VoxCPM2 inventory")
        #expect(try await store.inspect(model: model, artifact: artifact).reason == describe(rejection))
        await #expect(throws: rejection) { try await store.stagingDirectory(for: model, artifact: artifact) }
    }

    @Test("VDL-02 The import definition stays trusted and shares its identity and folder with the download definition")
    func importDefinition() async throws {
        let local = try VoxCPMModelArtifact.make()
        let download = try VoxCPMModelArtifact.makeDownload()
        try #require(local.download == nil && download.download != nil)
        let fixture = StoreFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let localModel = VoxCPMImportFixture.model(local)
        #expect(try await store.inspect(model: localModel, artifact: local).isMissing)
        let localArea = try await store.stagingDirectory(for: localModel, artifact: local)
        let downloadArea = try await store.stagingDirectory(
            for: VoxCPMImportFixture.model(download), artifact: download)
        #expect(localArea == downloadArea && localArea.lastPathComponent == local.sourceRevision)
        #expect(download.sourceKind == local.sourceKind && download.sourceRevision == local.sourceRevision)
    }

    @Test("VDL-03 An earlier import is ready under the download definition in its folder and ledger entry")
    func earlierImport() async throws {
        let fixture = try VoxCPMImportFixture()
        defer { fixture.files.remove() }
        let store = fixture.store()
        let installed = try await fixture.install(into: store)
        let ledger = try fixture.ledger()
        let artifact = try VoxCPMModelArtifact.makeDownload(from: fixture.artifact)
        try #require(artifact.download != nil && artifact.localBundle == fixture.artifact.localBundle)
        let model = VoxCPMImportFixture.model(artifact)
        let resource = try #require(try await store.inspect(model: model, artifact: artifact).resource)
        #expect(resource == installed && resource.identity == installed.identity)
        #expect(try fixture.ledger() == ledger)
        #expect(try fixture.files.ledger().entries.count == 1)
        #expect(fixture.stagingIsEmpty())
        let lease = try await store.acquire(
            model: model, artifact: artifact, directory: resource.directory, owner: UUID())
        #expect(await store.activeLeases(identity: resource.identity) == 1)
        await #expect(throws: ManagedModelStoreError.inUse) { try await store.delete(identity: resource.identity) }
        await lease.release()
        #expect(await store.activeLeases(identity: resource.identity) == 0)

        let demo = DemoDescriptor(
            id: "voice-design", title: "Voice design", systemImage: "person.wave.2", summary: "Voice design",
            platforms: [.macOS])
        let listed = LibraryModel(
            id: model.id, name: model.name, category: model.category, artifacts: [artifact],
            defaultArtifactID: artifact.id, platforms: [.macOS],
            demos: [ModelDemoLink(demoID: demo.id, platforms: [.macOS])])
        var state = ModelLibraryState(
            catalog: try ModelCatalog(models: [listed], demos: [demo]),
            environment: LibraryEnvironment(platform: .macOS), owner: UUID())
        _ = ModelLibraryFeature.reduce(&state, .refresh)
        let ticket = try #require(state.installations[artifact.id]?.requestID)
        #expect(ModelLibraryFeature.reduce(&state, .inspected(artifact.id, ticket, resource)).isEmpty)
        #expect(state.installations[artifact.id]?.phase == .ready)
        let ready = state
        #expect(ModelLibraryFeature.reduce(&state, .downloadRequested(artifact.id)).isEmpty)
        #expect(state == ready)
    }

    @Test(
        "VDL-03 An earlier import whose layout or sizes differ from the inventory is not ready to use",
        arguments: ["missing", "extra", "link", "size"])
    func mismatchedImport(_ kind: String) async throws {
        let fixture = try VoxCPMImportFixture()
        defer { fixture.files.remove() }
        let store = fixture.store()
        let installed = try await fixture.install(into: store)
        let expected: ManagedModelStoreError
        if kind == "size" {
            let file = installed.directory.appending(path: VoxCPMBundleInventory.tokenizerPath)
            try (try Data(contentsOf: file) + Data("x".utf8)).write(to: file)
            expected = .sizeMismatch(VoxCPMBundleInventory.tokenizerPath)
        } else {
            expected = try fixture.damage(kind, at: installed.directory)
        }
        let artifact = try VoxCPMModelArtifact.makeDownload(from: fixture.artifact)
        let inspection = try await store.inspect(model: VoxCPMImportFixture.model(artifact), artifact: artifact)
        #expect(inspection.resource == nil && inspection.reason == describe(expected))
    }

    @Test("VDL-03 An earlier import with changed contents of the same sizes passes the recorded check only")
    func sameSizeImport() async throws {
        let fixture = try VoxCPMImportFixture()
        defer { fixture.files.remove() }
        let store = fixture.store()
        let installed = try await fixture.install(into: store)
        let expected = try fixture.damage("altered", at: installed.directory)
        let artifact = try VoxCPMModelArtifact.makeDownload(from: fixture.artifact)
        let model = VoxCPMImportFixture.model(artifact)
        #expect(try await store.inspect(model: model, artifact: artifact).resource == installed)
        let full = try await store.inspect(model: model, artifact: artifact, verification: .full)
        #expect(full.resource == nil && full.reason == describe(expected))
    }
}
