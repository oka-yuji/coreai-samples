import ModelLibraryFeature
import ModelLibraryStorage
import Testing

struct ForecastCatalogTests {
    @Test("S3 Sales navigation shares the pinned Chronos artifact and keeps the NYC descriptor unchanged")
    func mappings() throws {
        let artifact = try ChronosModelArtifact.make()
        let catalogs = [try MacDemoCatalog.make(), try IOSDemoCatalog.make(chronosArtifact: artifact)]
        for catalog in catalogs {
            let model = try #require(catalog.model("chronos2"))
            #expect(model.artifacts == [artifact])
            #expect(model.demos.map(\.demoID) == ["taxi-forecast", "product-sales-forecast"])
            let sales = try #require(catalog.demo(ChronosDemoCatalog.salesSelection.demoID))
            #expect(sales.title == "Product sales forecast" && sales.allowsPreviewInteraction)
            #expect(sales.explanation.contains { $0.text.contains("CC BY-NC 3.0") })
            #expect(sales.explanation.contains { $0.text.contains("never starts") })
            let taxi = try #require(catalog.demo(ChronosDemoCatalog.selection.demoID))
            #expect(taxi.title == "NYC taxi pickup forecast" && taxi.allowsPreviewInteraction)
            #expect(
                taxi.summary
                    == "Explore fixed scenes with a precomputed preview, or select verified NYC API dates for an on-device forecast."
            )
            #expect(taxi.explanation.count == 3 && taxi.explanation[2].title == "Fixed-data preview")
        }
        #expect(try IOSDemoCatalog.make().demo("product-sales-forecast") == nil)
    }
}
