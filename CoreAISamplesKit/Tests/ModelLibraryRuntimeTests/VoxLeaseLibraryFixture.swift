import CryptoKit
import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryTransfer

@testable import ModelLibraryRuntime
@testable import ModelLibraryStorage

@MainActor
final class VoxLeaseWindow {
    let owner = UUID()
    let events = VoxLeaseSignals()
    private(set) var library: Store<ModelLibraryFeature>?
    private let fixture: VoxLeaseLibraryFixture
    init(fixture: VoxLeaseLibraryFixture) {
        self.fixture = fixture
        let runner = fixture.runner
        let observer = events
        let library = Store<ModelLibraryFeature>(
            initialState: ModelLibraryState(
                catalog: fixture.catalog, environment: LibraryEnvironment(platform: .macOS), owner: owner)
        ) { effect, send in
            await runner.run(effect, send: send)
            switch effect {
            case .inspect(_, let artifact, _): observer.record("inspect-" + artifact.id)
            case .acquireUsage: observer.record("usage-begin")
            case .releaseUsage: observer.record("usage-end")
            default: break
            }
        }
        self.library = library
        runner.register(library)
    }
    func current() throws -> Store<ModelLibraryFeature> {
        guard let library else { throw VoxLeaseProbeError(description: "window was released") }
        return library
    }
    func use(_ id: String) async throws -> ValidatedModelResource {
        let store = try current()
        let refreshes = fixture.model.artifacts.map { artifact in
            (artifact.id, events.log.count { $0 == "inspect-" + artifact.id })
        }
        store.send(.refresh)
        for (key, previous) in refreshes { try await events.wait("inspect-" + key, previous + 1) }
        store.send(.selectArtifact(modelID: fixture.model.id, artifactID: id))
        let n = events.log.count { $0 == "usage-begin" }
        store.send(.useRequested(fixture.model.id))
        try await events.wait("usage-begin", n + 1)
        guard let used = store.state.usages[fixture.model.id]?.resource,
            used.artifactID == id, used.registrationGeneration != nil
        else { throw VoxLeaseProbeError(description: "real window usage did not become ready") }
        return used
    }
    func end() { library?.send(.useEnded(fixture.model.id)) }
    func drop() { library = nil }
}

@MainActor
final class VoxLeaseLibraryFixture {
    static let old = "voxcpm2-fp16"
    static let new = "voxcpm2-reference_fp32"
    let base: URL
    let storage: ManagedModelStore
    let runner: ModelLibraryRunner
    let model: LibraryModel
    let catalog: ModelCatalog
    let runtimeUses = Store<ModelRuntimeUses>(initialState: ModelRuntimeUses.State()) { _, _ in }
    private var sources: [String: URL] = [:]

    init(manifests: URL, both: Bool) throws {
        let area = FileManager.default.temporaryDirectory.appending(path: "vox-managed-lease-" + UUID().uuidString)
        base = area
        try FileManager.default.createDirectory(at: area, withIntermediateDirectories: false)
        do {
            let legacy = try Self.payload("fp16", manifests: manifests, parent: area)
            let latest = both ? try Self.payload("reference_fp32", manifests: manifests, parent: area) : nil
            let a = try VoxCPMModelArtifact.make(inventory: legacy.inventory)
            var assets = [a]
            if let latest {
                assets.append(try VoxCPMModelArtifact.makeReference(inventory: latest.inventory))
                sources[Self.new] = latest.source
            }
            sources[Self.old] = legacy.source
            storage = ManagedModelStore(
                root: area.appending(path: "Models"), removeMigratedSources: false,
                clock: { Date(timeIntervalSince1970: 1000) },
                writeLedger: { try $0.write(to: $1, options: .atomic) }, availableCapacity: { _ in .max },
                voxcpmInventory: legacy.inventory, voxcpmReferenceInventory: latest?.inventory)
            let link = ModelDemoLink(demoID: "voice-design", platforms: [.macOS])
            model = LibraryModel(
                id: "voxcpm2", name: "VoxCPM2", category: "Speech", artifacts: assets,
                defaultArtifactID: a.id, platforms: [.macOS], demos: [link])
            let demo = DemoDescriptor(
                id: "voice-design", title: "Voice", systemImage: "mic", summary: "Local speech", platforms: [.macOS])
            catalog = try ModelCatalog(models: [model], demos: [demo])
            let transfer = ArtifactTransfer(
                fileLoader: { _, _ in throw VoxLeaseProbeError(description: "unexpected network") },
                availableCapacity: { _ in .max })
            runner = ModelLibraryRunner(
                resources: ModelLibraryResourceRunner(store: storage, transfer: transfer),
                registry: ModelDemoAdapterRegistry(), store: storage, retryDelays: [])
        } catch {
            try? FileManager.default.removeItem(at: area)
            throw error
        }
    }
    func cleanup() { try? FileManager.default.removeItem(at: base) }
    func staged(_ id: String) async throws -> (ModelArtifact, URL) {
        guard let asset = model.artifact(id), let source = sources[id] else {
            throw VoxLeaseProbeError(description: "unknown synthetic Vox profile")
        }
        return (asset, try await storage.copyForImport(from: source, model: model, artifact: asset) { _ in })
    }
    @discardableResult func install(_ id: String) async throws -> ValidatedModelResource {
        let (asset, stage) = try await staged(id)
        return try await storage.promote(staging: stage, model: model, artifact: asset)
    }
    func refuseDelete(_ identity: ModelArtifactIdentity) async throws {
        do { _ = try await storage.delete(identity: identity) } catch ManagedModelStoreError.inUse { return }
        throw VoxLeaseProbeError(description: "physical deletion succeeded during model use")
    }
    func refuseReplacement(_ stage: (ModelArtifact, URL)) async throws {
        do {
            _ = try await storage.promote(staging: stage.1, model: model, artifact: stage.0)
        } catch ManagedModelStoreError.inUse { return }
        throw VoxLeaseProbeError(description: "physical replacement succeeded during model use")
    }
    func counts(_ resource: ValidatedModelResource, physical: Int, displayed: Int) async throws {
        let real = await storage.activeLeases(identity: resource.identity)
        try requireLease(real == physical, "physical leases expected \(physical), found \(real)")
        let shown = runtimeUses.state.counts[model.id, default: 0]
        try requireLease(shown == displayed, "execution count expected \(displayed), found \(shown)")
    }

    private static func payload(_ profile: String, manifests: URL, parent: URL) throws -> (inventory: Data, source: URL)
    {
        let raw = try Data(contentsOf: manifests.appending(path: "voxcpm2-" + profile + "-manifest.json"))
        let expected =
            profile == "fp16"
            ? "fbe90a07bcdb4736b2b8b9dcd0b1f11b71857c28e4d506aad216bcda6c76754c"
            : "5beadfe31f8b37667f3e4ef42a8451b29dca1814a4cb53a253e96d9a2fdd1b51"
        try requireLease(digest(raw) == expected, "untrusted fixture manifest")
        guard let obj = try JSONSerialization.jsonObject(with: raw) as? [String: Any],
            let assets = obj["assets"] as? [String: [String: Any]]
        else {
            throw VoxLeaseProbeError(description: "fixture asset names")
        }
        var bytes = [
            "manifest.json": raw, "tokenizer.json": Data("fake tokenizer".utf8), "NOTICE": Data("notice".utf8),
            "LICENSE": Data("license".utf8),
        ]
        for (_, row) in assets {
            guard let path = row["path"] as? String else {
                throw VoxLeaseProbeError(description: "missing fixture path")
            }
            for name in ["main.mlirb", "main.hash", "metadata.json"] {
                bytes[path + "/" + name] = Data(("fake " + path + "/" + name).utf8)
            }
        }
        let source = parent.appending(path: profile, directoryHint: .isDirectory)
        let rows: [[String: Any]] = try bytes.sorted { $0.key < $1.key }.map { path, data in
            let url = source.appending(path: path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url, options: .withoutOverwriting)
            return ["path": path, "sha256": digest(data), "bytes": data.count]
        }
        try requireLease(rows.count == (profile == "fp16" ? 22 : 25), "fixture inventory count")
        let inventory = try JSONSerialization.data(
            withJSONObject: [
                "format": 1, "count": rows.count,
                "bytes": bytes.values.reduce(0) { $0 + $1.count }, "files": rows,
            ], options: .sortedKeys)
        return (inventory, source)
    }
    private static func digest(_ bytes: Data) -> String {
        SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }
}
