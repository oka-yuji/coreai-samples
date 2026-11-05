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

    @Test("LIB-01 The Mac catalog validates and lists seven models with their demos in declaration order on the Mac")
    func sections() throws {
        let catalog = try MacDemoCatalog.make()
        let sections = catalog.sections(platform: .macOS, audience: .release)
        #expect(
            sections.map(\.model.name) == [
                "Qwen3.8 27B", "Clef-Flash 9B", "EmbeddingGemma 2", "GLiNER2.5-multi", "Irodori", "TeleOCR", "JEV-27B",
            ])
        #expect(
            sections.map(\.id) == [
                "qwen3.8-27b", "clef-flash-9b", "embeddinggemma2", "gliner2.5-multi", "irodori", "teleocr", "jev-27b",
            ])
        #expect(
            sections.map { $0.demos.map(\.id) } == [
                ["chat"], ["decide", "recipes"], ["image-search", "video-search"], ["comparison"], ["speech"], ["ocr"],
                ["receipt-desk"],
            ])
        #expect(sections.map { $0.artifacts.count } == [1, 1, 2, 1, 1, 1, 1])
        #expect(catalog.sections(platform: .iOS, audience: .release).isEmpty)
        #expect(catalog.models.flatMap(\.artifacts).allSatisfy { $0.platforms == [.macOS] })
        #expect(catalog.demos.allSatisfy { $0.platforms == [.macOS] && $0.title == $0.summary })
        try catalog.validateConnections(adapterIDs: ["ocr", "receipt-desk"], unconnectedDemoIDs: unconnected)
        #expect(throws: ModelCatalogError.invalidDemo("speech")) {
            try catalog.validateConnections(
                adapterIDs: ["ocr", "receipt-desk"], unconnectedDemoIDs: unconnected.subtracting(["speech"]))
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
    }

    @Test(
        "LIB-01 The Mac self-test reports the Mac catalog with seven models, nine demos and the connected OCR and receipt demos"
    )
    func selfTest() async throws {
        let root = URL.temporaryDirectory.appending(path: "mac-catalog-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = ModelDemoAdapterRegistry()
        let (_, sink) = AsyncStream.makeStream(of: String.self)
        registry.register(FakeAdapter("ocr", label: OwnerLabels(), sink: sink))
        registry.register(FakeAdapter("receipt-desk", label: OwnerLabels(), sink: sink))
        let output = root.appending(path: "report.json")
        let status = await ModelLibrarySelfTest.run(
            output: output, root: root.appending(path: "managed"), registry: registry, unconnectedDemoIDs: unconnected,
            platform: .macOS, catalog: { try MacDemoCatalog.make() })
        #expect(status == 0)
        let report = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: output)) as? [String: Any])
        #expect(report["failures"] as? [String] == [] && report["catalogValidated"] as? Bool == true)
        #expect(report["visibleModels"] as? Int == 7 && report["visibleDemos"] as? Int == 9)
        #expect(report["catalogDemos"] as? [String] == unconnected.union(["ocr", "receipt-desk"]).sorted())
        let connections = try #require(report["registry"] as? [String: Any])
        #expect(connections["connectedDemos"] as? [String] == ["ocr", "receipt-desk"])
        #expect(connections["unconnectedDemos"] as? [String] == unconnected.sorted())
        #expect(connections["matchesCatalog"] as? Bool == true)
        #expect(report["capabilities"] == nil)
        #expect(report["scope"] as? String == "Catalog, registry and store checks of the Mac app.")
        let artifacts = try #require(report["artifacts"] as? [[String: Any]])
        #expect(artifacts.count == 8 && artifacts.allSatisfy { $0["inspection"] as? String == "missing" })
    }
}
