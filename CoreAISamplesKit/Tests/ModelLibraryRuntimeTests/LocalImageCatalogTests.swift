import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTestSupport
import ModelLibraryTransfer
import SwiftUI
import Testing

@testable import ModelLibraryRuntime

@Suite(.timeLimit(.minutes(1)))
@MainActor
struct LocalImageCatalogTests {
    @Test(
        "QMI-06 The ordinary catalog and acquisition harness accept an explicitly unconnected image demo without model callbacks"
    )
    func localCatalogAcquisition() async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let catalog = try IOSDemoCatalog.make(localImageArtifacts: [fixture.artifact])
        let adapters = catalog.demos.filter { $0.id != "image-generation" }.map { NoModelAdapter($0.id) }
        try catalog.validateConnections(adapterIDs: adapters.map(\.demoID), unconnectedDemoIDs: ["image-generation"])
        let assembly = ModelLibraryAssembly(adapters: adapters, store: fixture.store(), transfer: ArtifactTransfer())
        let result = try await ModelLibrarySelfTest.check(
            catalog: catalog, root: fixture.root, registry: assembly.registry, capabilities: [:],
            acquire: fixture.artifact.id, assembly: assembly, importBundle: fixture.source,
            unconnectedDemoIDs: ["image-generation"])
        #expect(result.failures.isEmpty)
        #expect(result.registry?.matchesCatalog == true)
        #expect(result.registry?.unconnectedDemos == ["image-generation"])
        let acquisition = try #require(result.acquisition?.artifacts.first)
        #expect(acquisition.route == "local-import")
        #expect(acquisition.state == "ready" && acquisition.inspection == "ready")
        #expect(acquisition.identity?.sourceKind == .localManifest)
        #expect(acquisition.revision == fixture.artifact.localBundle?.manifestSHA256)
        #expect(adapters.allSatisfy { $0.callbacks == 0 })
        let reopened = try await ModelLibrarySelfTest.check(
            catalog: catalog, root: fixture.root, registry: assembly.registry, capabilities: [:],
            unconnectedDemoIDs: ["image-generation"])
        #expect(reopened.failures.isEmpty)
        #expect(reopened.artifacts.first(where: { $0.artifactID == fixture.artifact.id })?.inspection == "ready")
    }

    @Test("QMI-06 A rejected replacement remains a failed acquisition even when the previous bundle stays ready")
    func rejectedReplacement() async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let store = fixture.store()
        let installed = try await fixture.install(into: store)
        let catalog = try IOSDemoCatalog.make(localImageArtifacts: [fixture.artifact])
        let adapters = catalog.demos.filter { $0.id != "image-generation" }.map { NoModelAdapter($0.id) }
        let assembly = ModelLibraryAssembly(adapters: adapters, store: store, transfer: ArtifactTransfer())
        try Data("altered".utf8).write(to: fixture.source.appending(path: "bundle.json"))
        let result = try await ModelLibrarySelfTest.check(
            catalog: catalog, root: fixture.root, registry: assembly.registry, capabilities: [:],
            acquire: fixture.artifact.id, assembly: assembly, importBundle: fixture.source,
            unconnectedDemoIDs: ["image-generation"])
        let acquisition = try #require(result.acquisition?.artifacts.first)
        #expect(acquisition.state == "ready" && acquisition.inspection == "ready")
        #expect(acquisition.failure != nil)
        #expect(result.failures.count == 1)
        #expect(acquisition.identity?.revision == installed.revision)
        #expect(adapters.allSatisfy { $0.callbacks == 0 })
    }

    @Test("QMI-06 A local import requires one exact catalog target before acquisition begins")
    func exactLocalTarget() async throws {
        let fixture = try ImageBundleFixture()
        defer { fixture.remove() }
        let catalog = try IOSDemoCatalog.make(localImageArtifacts: [fixture.artifact])
        for target: String? in [nil, "all", "embeddinggemma2-fp16-safe-s128-v70"] {
            await #expect(throws: (any Error).self) {
                try await ModelLibrarySelfTest.check(
                    catalog: catalog, root: fixture.root, registry: nil, capabilities: nil,
                    acquire: target, importBundle: fixture.source)
            }
        }
        #expect(!FileManager.default.fileExists(atPath: fixture.root.appending(path: "artifacts").path))
    }
}

@MainActor
private final class NoModelAdapter: ModelDemoAdapter {
    let demoID: String
    var callbacks = 0

    init(_ demoID: String) { self.demoID = demoID }

    func preview(for selection: DemoSelection) -> AnyView {
        callbacks += 1
        return AnyView(EmptyView())
    }

    func content(for session: ModelDemoSession, gate: DemoExecutionGate) -> AnyView {
        callbacks += 1
        return AnyView(EmptyView())
    }

    func stop(_ session: ModelDemoSession) async throws { callbacks += 1 }
}
