import FeatureStore
import Foundation
import ModelLibraryRuntime
import ModelLibraryStorage
import Testing

@testable import ModelLibraryFeature

@MainActor
struct MacDemoCatalogTests {
    private let unconnected: Set<String> = [
        "chat", "decide", "recipes", "image-search", "video-search", "comparison", "speech",
    ]

    @Test("LIB-01 The Mac catalog validates and lists eight models with their demos in declaration order on the Mac")
    func sections() throws {
        let catalog = try MacDemoCatalog.make()
        let sections = catalog.sections(platform: .macOS, audience: .release)
        #expect(
            sections.map(\.model.name) == [
                "Qwen3.8 27B", "Clef-Flash 9B", "EmbeddingGemma 2", "GLiNER2.5-multi", "Irodori", "TeleOCR", "JEV-27B",
                "Chronos-2",
            ])
        #expect(
            sections.map(\.id) == [
                "qwen3.8-27b", "clef-flash-9b", "embeddinggemma2", "gliner2.5-multi", "irodori", "teleocr", "jev-27b",
                "chronos2",
            ])
        #expect(
            sections.map { $0.demos.map(\.id) } == [
                ["chat"], ["decide", "recipes"], ["image-search", "video-search"], ["comparison"], ["speech"], ["ocr"],
                ["receipt-desk"], ["taxi-forecast", "product-sales-forecast", "seoul-bike-forecast"],
            ])
        #expect(sections.map { $0.artifacts.count } == [1, 1, 2, 1, 1, 1, 1, 1])
        #expect(catalog.sections(platform: .iOS, audience: .release).isEmpty)
        #expect(
            catalog.models.filter { $0.id != "chronos2" }.flatMap(\.artifacts).allSatisfy { $0.platforms == [.macOS] })
        #expect(catalog.demos.allSatisfy { $0.platforms == [.macOS] })
        #expect(
            catalog.demos.filter {
                !["taxi-forecast", "product-sales-forecast", "seoul-bike-forecast"].contains($0.id)
            }.allSatisfy {
                $0.title == $0.summary
            })
        try catalog.validateConnections(
            adapterIDs: ["ocr", "product-sales-forecast", "receipt-desk", "seoul-bike-forecast", "taxi-forecast"],
            unconnectedDemoIDs: unconnected)
        #expect(throws: ModelCatalogError.invalidDemo("speech")) {
            try catalog.validateConnections(
                adapterIDs: ["ocr", "product-sales-forecast", "receipt-desk", "seoul-bike-forecast", "taxi-forecast"],
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
        #expect(try catalog.model("teleocr")?.artifacts == [TeleOCRModelArtifact.make()])
        #expect(try catalog.model("chronos2")?.artifacts == [ChronosModelArtifact.make()])
    }

    @Test(
        "LIB-01 The Mac self-test reports eight models, twelve demos and the connected OCR, receipt and Chronos demos"
    )
    func selfTest() async throws {
        let root = URL.temporaryDirectory.appending(path: "mac-catalog-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = ModelDemoAdapterRegistry()
        let (_, sink) = AsyncStream.makeStream(of: String.self)
        registry.register(FakeAdapter("ocr", label: OwnerLabels(), sink: sink))
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
        #expect(report["visibleModels"] as? Int == 8 && report["visibleDemos"] as? Int == 12)
        #expect(
            report["catalogDemos"] as? [String]
                == unconnected.union([
                    "ocr", "product-sales-forecast", "receipt-desk", "seoul-bike-forecast", "taxi-forecast",
                ]).sorted())
        let connections = try #require(report["registry"] as? [String: Any])
        #expect(
            connections["connectedDemos"] as? [String] == [
                "ocr", "product-sales-forecast", "receipt-desk", "seoul-bike-forecast", "taxi-forecast",
            ])
        #expect(connections["unconnectedDemos"] as? [String] == unconnected.sorted())
        #expect(connections["matchesCatalog"] as? Bool == true)
        #expect(report["capabilities"] == nil)
        #expect(report["scope"] as? String == "Catalog, registry and store checks of the Mac app.")
        let artifacts = try #require(report["artifacts"] as? [[String: Any]])
        #expect(artifacts.count == 9 && artifacts.allSatisfy { $0["inspection"] as? String == "missing" })
    }
}
