import Foundation
import ModelLibraryFeature
import Testing

struct ModelVariantDisplayTests {
    private let fixture = LibraryFixture()
    private let english = Locale(identifier: "en_US")

    private func reduce(
        _ state: inout ModelLibraryState, _ action: ModelLibraryAction
    ) -> [ModelLibraryFeature.Operation] {
        ModelLibraryFeature.reduce(&state, action)
    }

    private func displayCatalog() throws -> ModelCatalog {
        let described = ModelArtifact(id: "docs", title: "Docs", contract: "docs", platforms: [.iOS])
        let listed = ["a", "b"].map {
            ModelArtifact(id: "list-" + $0, title: $0.uppercased(), contract: "list", platforms: [.iOS])
        }
        let unselected = [
            fixture.artifact("fp16"), fixture.artifact("int8"), fixture.artifact("int4", visibility: .development),
        ]
        return try ModelCatalog(
            models: [
                fixture.model("single", artifacts: [fixture.artifact("fp32")], defaultArtifactID: "fp32"),
                fixture.model("described", artifacts: [described], defaultArtifactID: "docs"),
                fixture.model("listed", artifacts: listed, defaultArtifactID: "list-a"),
                fixture.model("unselected", artifacts: unselected, defaultArtifactID: "int4"),
            ],
            demos: [fixture.demo("text")])
    }

    @Test(
        "SHOW-01 The demo host heading names the selected variant of a model with several visible variants and updates when the variant changes"
    )
    func headingFollowsVariant() throws {
        var state = fixture.state(try fixture.variantCatalog(), audience: .development)
        #expect(state.variantTitle(for: "model") == "FP32")
        #expect(reduce(&state, .selectArtifact(modelID: "model", artifactID: "int4")).isEmpty)
        #expect(state.variantTitle(for: "model") == "INT4")

        _ = reduce(&state, .openDemo(DemoSelection(modelID: "model", demoID: "text")))
        try fixture.install(&state, "int4")
        let running = try #require(state.session)
        _ = reduce(&state, .selectArtifact(modelID: "model", artifactID: "fp32"))
        let stop = try #require(state.stopRequestID)
        #expect(state.variantTitle(for: "model") == "INT4")
        #expect(reduce(&state, .demoStopped(running.id, stop)).isEmpty)
        #expect(state.variantTitle(for: "model") == "FP32")
    }

    @Test(
        "SHOW-01 Several visible variants are always named, a single variant only when it has a download, and none without a visible selection"
    )
    func headingNamesDistinguishableVariant() throws {
        let state = fixture.state(try displayCatalog())
        #expect(state.variantTitle(for: "listed") == "A")
        #expect(state.variantTitle(for: "single") == "FP32")
        #expect(state.variantTitle(for: "described") == nil)
        #expect(state.artifacts(for: "unselected").count == 2 && state.selectedArtifact(for: "unselected") == nil)
        #expect(state.variantTitle(for: "unselected") == nil)
        #expect(state.variantTitle(for: "unknown") == nil)
    }

    @Test("SHOW-02 Each Models row shows the display name and size of the selected variant")
    func rowShowsSelectedVariant() throws {
        var state = fixture.state(try fixture.variantCatalog(), audience: .development)
        let model = try #require(state.catalog.model("model"))
        #expect(state.rowDetail(for: model, locale: english) == "Fixture · FP32 · 4 bytes")
        #expect(state.rowDetail(for: model, locale: Locale(identifier: "de_DE")) == "Fixture · FP32 · 4 Byte")
        _ = reduce(&state, .refresh)
        let request = try #require(state.installations["int4"]?.requestID)
        let installed = ValidatedModelResource(
            modelID: "model", artifactID: "int4", contract: "fixture-int4",
            revision: LibraryFixture.distributionRevision,
            upstreamRevision: LibraryFixture.upstreamRevision, directory: URL(filePath: "/fixture/int4"),
            sizeBytes: 2_000_000)
        #expect(reduce(&state, .inspected("int4", request, installed)).isEmpty)
        #expect(state.rowDetail(for: model, locale: english) == "Fixture · FP32 · 4 bytes")
        #expect(reduce(&state, .selectArtifact(modelID: "model", artifactID: "int4")).isEmpty)
        #expect(state.rowDetail(for: model, locale: english) == "Fixture · INT4 · 2 MB")

        let rows = fixture.state(try displayCatalog())
        let single = try #require(rows.catalog.model("single"))
        let described = try #require(rows.catalog.model("described"))
        let unselected = try #require(rows.catalog.model("unselected"))
        #expect(rows.rowDetail(for: single, locale: english) == "Fixture · FP32 · 4 bytes")
        #expect(rows.rowDetail(for: described, locale: english) == "Fixture")
        #expect(rows.rowDetail(for: unselected, locale: english) == "Fixture")
    }
}
