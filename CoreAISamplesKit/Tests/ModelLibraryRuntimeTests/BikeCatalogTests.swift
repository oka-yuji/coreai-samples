import ModelLibraryFeature
import ModelLibraryStorage
import Testing

struct BikeCatalogTests {
    @Test("B3 Bike navigation shares the Chronos artifact and exposes offline preview without removing sales or NYC")
    func mappings() throws {
        let artifact = try ChronosModelArtifact.make()
        for catalog in [try MacDemoCatalog.make(), try IOSDemoCatalog.make(chronosArtifact: artifact)] {
            let model = try #require(catalog.model("chronos2"))
            #expect(model.artifacts == [artifact])
            let link = try #require(model.demos.first { $0.demoID == ChronosDemoCatalog.bikeSelection.demoID })
            #expect(link.optionalCapabilities.isEmpty)
            let demo = try #require(catalog.demo(link.demoID))
            #expect(demo.title == "Seoul Bike forecast" && demo.allowsPreviewInteraction)
            #expect(demo.explanation.contains { $0.text.contains("not model input") })
            #expect(demo.explanation.contains { $0.text.contains("CC BY 4.0") })
            #expect(catalog.demo(ChronosDemoCatalog.salesSelection.demoID) != nil)
            #expect(catalog.demo(ChronosDemoCatalog.selection.demoID) != nil)
        }
        #expect(try IOSDemoCatalog.make().demo(ChronosDemoCatalog.bikeSelection.demoID) == nil)
    }
}
