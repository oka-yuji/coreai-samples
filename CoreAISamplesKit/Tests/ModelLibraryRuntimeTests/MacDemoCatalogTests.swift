import FeatureStore
import Foundation
import ModelLibraryRuntime
import ModelLibraryStorage
import Testing

@testable import ModelLibraryFeature

@MainActor
struct MacDemoCatalogTests {
    private let unconnected: Set<String> = [
        "chat", "decide", "recipes", "image-search", "video-search", "comparison", "speech", "voice-design",
    ]

    @Test("LIB-01 The Mac catalog validates and lists nine models with their demos in declaration order on the Mac")
    func sections() throws {
        let catalog = try MacDemoCatalog.make()
        let sections = catalog.sections(platform: .macOS, audience: .release)
        #expect(
            sections.map(\.model.name) == [
                "Qwen3.8 27B", "Clef-Flash 9B", "EmbeddingGemma 2", "GLiNER2.5-multi", "Irodori", "VoxCPM2", "TeleOCR",
                "JEV-27B", "Chronos-2",
            ])
        #expect(
            sections.map(\.id) == [
                "qwen3.8-27b", "clef-flash-9b", "embeddinggemma2", "gliner2.5-multi", "irodori", "voxcpm2", "teleocr",
                "jev-27b", "chronos2",
            ])
        #expect(
            sections.map { $0.demos.map(\.id) } == [
                ["chat"], ["decide", "recipes"], ["image-search", "video-search", "spoken-search"], ["comparison"],
                ["speech"],
                ["voice-design"], ["ocr"], ["receipt-desk"],
                ["taxi-forecast", "product-sales-forecast", "seoul-bike-forecast"],
            ])
        #expect(sections.map { $0.artifacts.count } == [1, 1, 2, 1, 1, 1, 1, 1, 1])
        #expect(catalog.sections(platform: .iOS, audience: .release).isEmpty)
        #expect(
            catalog.models.filter { $0.id != "chronos2" }.flatMap(\.artifacts).allSatisfy { $0.platforms == [.macOS] })
        #expect(catalog.demos.allSatisfy { $0.platforms == [.macOS] })
        #expect(
            catalog.demos.filter {
                !["taxi-forecast", "product-sales-forecast", "seoul-bike-forecast", "spoken-search"].contains($0.id)
            }.allSatisfy {
                $0.title == $0.summary
            })
        try catalog.validateConnections(
            adapterIDs: [
                "ocr", "product-sales-forecast", "receipt-desk", "seoul-bike-forecast", "spoken-search",
                "taxi-forecast",
            ],
            unconnectedDemoIDs: unconnected)
        #expect(throws: ModelCatalogError.invalidDemo("speech")) {
            try catalog.validateConnections(
                adapterIDs: [
                    "ocr", "product-sales-forecast", "receipt-desk", "seoul-bike-forecast", "spoken-search",
                    "taxi-forecast",
                ],
                unconnectedDemoIDs: unconnected.subtracting(["speech"]))
        }
        for id in ["qwen3.8-27b", "clef-flash-9b"] {
            let artifacts = try #require(catalog.model(id)).artifacts
            #expect(artifacts.count == 1 && artifacts.allSatisfy { $0.sourceKind == .huggingFace && !$0.files.isEmpty })
        }
        #expect(
            catalog.model("gliner2.5-multi")?.artifacts == [PublishedModelArtifacts.glinerFP32(platforms: [.macOS])]
        )
        #expect(
            catalog.model("embeddinggemma2")?.artifacts == [
                PublishedModelArtifacts.embeddingFP16(platforms: [.macOS]),
                PublishedModelArtifacts.embeddingPTQ(platforms: [.macOS]),
            ])
        #expect(
            try catalog.model("irodori")?.artifacts == [IrodoriModelArtifact.makeReferenceDownload(platforms: [.macOS])]
        )
        #expect(try catalog.model("voxcpm2")?.artifacts == [VoxCPMModelArtifact.makeDownload()])
        #expect(try catalog.model("teleocr")?.artifacts == [TeleOCRModelArtifact.make()])
        #expect(try catalog.model("chronos2")?.artifacts == [ChronosModelArtifact.make()])
    }

    @Test("AUD-04 Spoken word search opens from either release catalog with the embedding models missing")
    func spokenSearch() throws {
        let aliases = ["spoken-search": ["Spoken word alias"]]
        let catalogs: [(ModelCatalog, LibraryPlatform)] = [
            (try MacDemoCatalog.make(searchTerms: aliases), .macOS),
            (try IOSDemoCatalog.make(searchTerms: aliases), .iOS),
        ]
        let selection = SpokenSearchCatalog.selection
        for (catalog, platform) in catalogs {
            let matches = catalog.sections(platform: platform, audience: .release, query: "spoken word alias")
            #expect(matches.map(\.id) == ["embeddinggemma2"])
            #expect(matches.first?.demos.map(\.id) == [selection.demoID])
            let descriptor = try #require(catalog.demo(selection.demoID))
            #expect(descriptor.allowsPreviewInteraction && descriptor.title == SpokenSearchCatalog.title)
            let link = try #require(catalog.model(selection.modelID)?.demos.first { $0.demoID == selection.demoID })
            #expect(link.requiredCapabilities.isEmpty)
            var state = ModelLibraryState(
                catalog: catalog, environment: LibraryEnvironment(platform: platform), owner: UUID())
            #expect(ModelLibraryFeature.reduce(&state, .openDemo(selection)).isEmpty)
            #expect(state.selection == selection && state.access(to: selection) == .preview(.missing))
        }
    }

    @Test("AUD-04 Spoken word search can use a recording while another window holds the model")
    func spokenSearchWhileOccupied() throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let catalog = try MacDemoCatalog.make()
        let selection = SpokenSearchCatalog.selection
        let artifact = try #require(catalog.model(selection.modelID)?.artifacts.first)
        let adapter = FakeAdapter(selection.demoID, label: harness.labels, sink: harness.sink)
        harness.registry.register(adapter)
        let environment = LibraryEnvironment(platform: .macOS, connectedDemos: [selection.demoID])
        var state = ModelLibraryState(catalog: catalog, environment: environment, owner: UUID())
        state.installations[artifact.id]?.phase = .ready
        state.installations[artifact.id]?.resource = ValidatedModelResource(
            modelID: selection.modelID, artifactID: artifact.id, contract: artifact.contract,
            revision: try #require(artifact.sourceRevision), upstreamRevision: artifact.upstream?.revision,
            directory: harness.base, sizeBytes: artifact.sizeBytes ?? 0)
        _ = ModelLibraryFeature.reduce(&state, .openDemo(selection))
        let session = try #require(state.session)
        _ = ModelLibraryFeature.reduce(&state, .demoPreparationChanged(session.id, .occupied))
        let store = Store<ModelLibraryFeature>(initialState: state) { _, _ in }
        let rendering = ModelLibraryRendering(registry: harness.registry, runner: harness.runner, library: store)
        _ = rendering.content(for: session)
        #expect(adapter.previews == [selection] && adapter.contents.isEmpty)
        store.send(.closeDemo)
        _ = rendering.content(for: session)
        #expect(adapter.previews == [selection] && adapter.contents.isEmpty)
    }

    @Test("VUI-01 VoxCPM2 and its speech demo are listed on the Mac only, after Irodori, and not in the iOS catalog")
    func voxcpm2() throws {
        let catalog = try MacDemoCatalog.make()
        let model = try #require(catalog.model("voxcpm2"))
        #expect(model.name == "VoxCPM2" && model.category == "Speech Synthesis" && model.platforms == [.macOS])
        #expect(model.artifacts == [try VoxCPMModelArtifact.makeDownload()] && model.artifacts[0].platforms == [.macOS])
        #expect(model.demos == [ModelDemoLink(demoID: "voice-design", platforms: [.macOS])])
        let demo = try #require(catalog.demo("voice-design"))
        #expect(demo.platforms == [.macOS] && demo.title == demo.summary)
        let sections = catalog.sections(platform: .macOS, audience: .release)
        #expect(Array(sections.map(\.id).drop { $0 != "irodori" }.prefix(2)) == ["irodori", "voxcpm2"])
        #expect(sections.first { $0.id == "voxcpm2" }?.demos.map(\.id) == ["voice-design"])
        #expect(catalog.sections(platform: .iOS, audience: .release).isEmpty)
        #expect(throws: ModelCatalogError.invalidDemo("voice-design")) {
            try catalog.validateConnections(
                adapterIDs: [
                    "ocr", "product-sales-forecast", "receipt-desk", "seoul-bike-forecast", "spoken-search",
                    "taxi-forecast",
                ],
                unconnectedDemoIDs: unconnected.subtracting(["voice-design"]))
        }
        let phone = try IOSDemoCatalog.make(
            speechArtifact: IrodoriModelArtifact.make(),
            referenceSpeechArtifact: IrodoriModelArtifact.makeReferenceDownload(),
            chronosArtifact: ChronosModelArtifact.make())
        #expect(phone.model("voxcpm2") == nil && phone.demo("voice-design") == nil)
        #expect(phone.models.flatMap(\.artifacts).allSatisfy { $0.contract != VoxCPMModelArtifact.contract })
        #expect(phone.sections(platform: .iOS, audience: .release).allSatisfy { $0.id != "voxcpm2" })
    }

    @Test(
        "LIB-01 The Mac self-test reports nine models, fourteen demos and their connected adapters"
    )
    func selfTest() async throws {
        let root = URL.temporaryDirectory.appending(path: "mac-catalog-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = ModelDemoAdapterRegistry()
        let (_, sink) = AsyncStream.makeStream(of: String.self)
        registry.register(FakeAdapter("ocr", label: OwnerLabels(), sink: sink))
        registry.register(FakeAdapter("spoken-search", label: OwnerLabels(), sink: sink))
        registry.register(FakeAdapter("receipt-desk", label: OwnerLabels(), sink: sink))
        registry.register(FakeAdapter("taxi-forecast", label: OwnerLabels(), sink: sink))
        registry.register(FakeAdapter("product-sales-forecast", label: OwnerLabels(), sink: sink))
        registry.register(FakeAdapter("seoul-bike-forecast", label: OwnerLabels(), sink: sink))
        let output = root.appending(path: "report.json")
        let status = await ModelLibrarySelfTest.run(
            output: output, root: root.appending(path: "managed"), registry: registry, unconnectedDemoIDs: unconnected,
            platform: .macOS, catalog: { try MacDemoCatalog.make() })
        #expect(status == 0)
        let report = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: output)) as? [String: Any])
        #expect(report["failures"] as? [String] == [] && report["catalogValidated"] as? Bool == true)
        #expect(report["visibleModels"] as? Int == 9 && report["visibleDemos"] as? Int == 14)
        #expect(
            report["catalogDemos"] as? [String]
                == unconnected.union([
                    "ocr", "product-sales-forecast", "receipt-desk", "seoul-bike-forecast", "spoken-search",
                    "taxi-forecast",
                ]).sorted())
        let connections = try #require(report["registry"] as? [String: Any])
        #expect(
            connections["connectedDemos"] as? [String] == [
                "ocr", "product-sales-forecast", "receipt-desk", "seoul-bike-forecast", "spoken-search",
                "taxi-forecast",
            ])
        #expect(connections["unconnectedDemos"] as? [String] == unconnected.sorted())
        #expect(connections["matchesCatalog"] as? Bool == true)
        #expect(report["capabilities"] == nil)
        #expect(report["scope"] as? String == "Catalog, registry and store checks of the Mac app.")
        let artifacts = try #require(report["artifacts"] as? [[String: Any]])
        #expect(artifacts.count == 10 && artifacts.allSatisfy { $0["inspection"] as? String == "missing" })
    }
}
