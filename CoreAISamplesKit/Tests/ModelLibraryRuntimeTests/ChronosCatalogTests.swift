import Foundation
import ModelLibraryRuntime
import ModelLibraryStorage
import Testing

@testable import ModelLibraryFeature

@MainActor
struct ChronosCatalogTests {
    @Test("M1 Both platform catalogs expose the pinned Chronos identity and an explicit forecast preview")
    func catalogs() throws {
        let artifact = try ChronosModelArtifact.make()
        for (platform, catalog) in try catalogs(artifact) {
            let model = try #require(catalog.model("chronos2"))
            #expect(model.artifacts == [artifact] && model.defaultArtifactID == "chronos2-fp32")
            #expect(
                model.platforms == [platform]
                    && model.demos.map(\.demoID) == ["taxi-forecast", "product-sales-forecast"])
            #expect(
                model.demos.allSatisfy { $0.requiredCapabilities.isEmpty }
                    && model.demos[0].optionalCapabilities.isEmpty && model.demos[1].optionalCapabilities == [.network])
            let demo = try #require(catalog.demo("taxi-forecast"))
            #expect(demo.summary.contains("precomputed preview") && demo.summary.contains("on-device forecast"))
            #expect(demo.summary.contains("verified NYC API dates"))
            #expect(demo.explanation.contains { $0.text.contains("479 MB") && $0.text.contains("Import") })
            #expect(demo.explanation.contains { $0.text.contains("Download") && $0.text.contains("import") })
            #expect(!demo.explanation.contains { $0.text.contains("does not provide a model download") })
            #expect(demo.explanation.contains { $0.text.contains("does not start a prediction") })
            let connected = catalog.demos.map(\.id).filter { $0 != demo.id }
            try catalog.validateConnections(adapterIDs: connected, unconnectedDemoIDs: [demo.id])
            #expect(throws: ModelCatalogError.invalidDemo(demo.id)) {
                try catalog.validateConnections(adapterIDs: connected)
            }
            #expect(throws: (any Error).self) {
                try catalog.validateConnections(adapterIDs: [], unconnectedDemoIDs: [demo.id])
            }
            #expect(throws: ModelCatalogError.invalidDemo("unknown")) {
                try catalog.validateConnections(adapterIDs: connected, unconnectedDemoIDs: [demo.id, "unknown"])
            }
        }
    }

    @Test("M1 An iOS catalog without an injected Chronos artifact does not invent one")
    func optionalArtifact() throws {
        let catalog = try IOSDemoCatalog.make()
        #expect(catalog.model("chronos2") == nil && catalog.demo("taxi-forecast") == nil)
        let injected = try IOSDemoCatalog.make(
            searchTerms: ["chronos2": ["future observations"], "taxi-forecast": ["taxi demand"]],
            chronosArtifact: ChronosModelArtifact.make())
        #expect(
            injected.sections(platform: .iOS, audience: .release, query: "future observations").map(\.id) == [
                "chronos2"
            ])
        #expect(
            injected.sections(platform: .iOS, audience: .release, query: "taxi demand").first?.demos.map(\.id) == [
                "taxi-forecast"
            ])
    }

    @Test("M1 First use and a ready Chronos artifact remain previews without model preparation or a session")
    func firstUseAndReady() throws {
        let artifact = try ChronosModelArtifact.make()
        for (platform, catalog) in try catalogs(artifact) {
            let selection = ChronosDemoCatalog.selection
            let owner = UUID()
            var state = ModelLibraryState(
                catalog: catalog,
                environment: LibraryEnvironment(
                    platform: platform, connectedDemos: Set(catalog.demos.map(\.id)).subtracting([selection.demoID])),
                owner: owner)
            #expect(state.session == nil && !state.isConnected(selection))
            #expect(ModelLibraryFeature.reduce(&state, .openDemo(selection)).isEmpty)
            #expect(state.access(to: selection) == .preview(.missing))
            _ = ModelLibraryFeature.reduce(&state, .refresh)
            let ticket = try #require(state.installations[artifact.id]?.requestID)
            let resource = ValidatedModelResource(
                modelID: selection.modelID, artifactID: artifact.id, contract: artifact.contract,
                revision: try #require(artifact.sourceRevision), upstreamRevision: artifact.upstream?.revision,
                directory: URL.temporaryDirectory.appending(path: "chronos-catalog-\(owner)"),
                sizeBytes: try #require(artifact.totalFileBytes), sourceKind: .localManifest)
            #expect(ModelLibraryFeature.reduce(&state, .inspected(artifact.id, ticket, resource)).isEmpty)
            #expect(state.installations[artifact.id]?.phase == .ready)
            #expect(state.access(to: selection) == .preview(.ready) && state.session == nil)
            #expect(ModelLibraryFeature.reduce(&state, .requestDemoPreparation(ticket)).isEmpty)
            #expect(ModelLibraryFeature.reduce(&state, .closeDemo).isEmpty)
            #expect(ModelLibraryFeature.reduce(&state, .openDemo(selection)).isEmpty)
            #expect(state.access(to: selection) == .preview(.ready) && state.session == nil)
        }
    }

    private func catalogs(_ artifact: ModelArtifact) throws -> [(LibraryPlatform, ModelCatalog)] {
        [(.macOS, try MacDemoCatalog.make()), (.iOS, try IOSDemoCatalog.make(chronosArtifact: artifact))]
    }
}

extension ChronosCatalogTests {
    @Test("D4 A connected Chronos demo previews missing data and creates a session only for a ready model")
    func connectedChronos() throws {
        let artifact = try ChronosModelArtifact.make()
        let catalog = try MacDemoCatalog.make()
        let selection = ChronosDemoCatalog.selection
        var state = ModelLibraryState(
            catalog: catalog, environment: LibraryEnvironment(platform: .macOS, connectedDemos: [selection.demoID]),
            owner: UUID())
        _ = ModelLibraryFeature.reduce(&state, .openDemo(selection))
        #expect(state.access(to: selection) == .preview(.missing) && state.session == nil)
        _ = ModelLibraryFeature.reduce(&state, .refresh)
        let ticket = try #require(state.installations[artifact.id]?.requestID)
        let resource = ValidatedModelResource(
            modelID: selection.modelID, artifactID: artifact.id, contract: artifact.contract,
            revision: try #require(artifact.sourceRevision), upstreamRevision: artifact.upstream?.revision,
            directory: URL(filePath: "/unused-chronos"), sizeBytes: try #require(artifact.totalFileBytes),
            sourceKind: .localManifest)
        _ = ModelLibraryFeature.reduce(&state, .inspected(artifact.id, ticket, resource))
        #expect(state.installations[artifact.id]?.phase == .ready)
        _ = ModelLibraryFeature.reduce(&state, .openDemo(selection))
        let session = try #require(state.session)
        #expect(state.access(to: selection) == .ready(session) && state.demoPreparation == .pending)
        let commands = ModelLibraryFeature.reduce(&state, .requestDemoPreparation(session.id))
        #expect(!commands.isEmpty && state.demoPreparation == .preparing)
    }
}
