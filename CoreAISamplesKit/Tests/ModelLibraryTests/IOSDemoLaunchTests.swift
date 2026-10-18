import FeatureStore
import Foundation
import ModelLibraryUI
import Testing

@testable import ModelLibraryFeature

struct IOSDemoLaunchTests {
    @Test(
        "Ordinary launch opens Models and explicit diagnostics preserve their destinations",
        arguments: [
            ([String](), IOSDemoLaunch.models),
            (["app", "--cpu"], .models),
            (["app", "--extract-eval"], .models),
            (["app", "--extract-eval", "fixture"], .extractionEvaluation),
            (["app", "--extract-eval=fixture"], .extractionEvaluation),
            (["app", "--image-search-eval", "fixture"], .imageSearchEvaluation),
            (["app", "--image-search-eval", "--extract-eval", "fixture"], .imageSearchEvaluation),
            (["app", "--image-search"], .legacyDemos),
            (["app", "--object-finder"], .legacyDemos),
            (["app", "--truck-inspection"], .legacyDemos),
            (["app", "--legacy-demos"], .legacyDemos),
        ])
    func destination(arguments: [String], expected: IOSDemoLaunch) {
        #expect(IOSDemoLaunch.resolve(arguments: arguments) == expected)
    }

    @Test("The iOS catalog presents two models and five demos without pretending their downloads are configured")
    func catalog() throws {
        let catalog = try IOSDemoCatalog.make()
        let sections = catalog.sections(platform: .iOS, audience: .release)
        #expect(sections.map(\.model.name) == ["GLiNER2.5-multi", "EmbeddingGemma 2"])
        #expect(
            sections.map { $0.demos.map(\.id) } == [
                ["comparison", "extraction", "inspection"], ["image-search", "object-finder"],
            ])
        #expect(
            catalog.models.flatMap(\.artifacts).allSatisfy {
                $0.download == nil && $0.sizeBytes == nil && $0.files.isEmpty && $0.upstream == nil
            })
        #expect(catalog.sections(platform: .macOS, audience: .release).isEmpty)
    }

    @Test("A translated demo search retains its model while filtering unrelated demos")
    func localizedSearch() throws {
        let catalog = try IOSDemoCatalog.make(searchTerms: ["comparison": ["comparar productos"]])
        let matches = catalog.sections(platform: .iOS, audience: .release, query: "comparar")
        #expect(matches.map(\.id) == ["gliner2.5-multi"])
        #expect(matches.first?.demos.map(\.id) == ["comparison"])
        #expect(catalog.sections(platform: .iOS, audience: .release, query: "embeddinggemma").first?.demos.count == 2)
    }

    @Test("Demo descriptors encoded before search aliases were added still decode")
    func decoding() throws {
        let descriptor = try #require(IOSDemoCatalog.make().demos.first)
        let data = try JSONEncoder().encode(descriptor)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["searchTerms"] == nil)
        #expect(try JSONDecoder().decode(DemoDescriptor.self, from: data) == descriptor)
    }

    @MainActor
    @Test("Browsing all production demo descriptions never creates an executable session")
    func browseWithoutResources() throws {
        let catalog = try IOSDemoCatalog.make()
        var state = ModelLibraryState(catalog: catalog, environment: LibraryEnvironment(platform: .iOS), owner: UUID())
        for command in ModelLibraryFeature.reduce(&state, .refresh) {
            if case .run(let effect, _) = command {
                ModelLibraryBrowsingRunner.run(effect) { action in
                    #expect(ModelLibraryFeature.reduce(&state, action).isEmpty)
                }
            }
        }
        for section in state.sections {
            for demo in section.demos {
                let selection = DemoSelection(modelID: section.id, demoID: demo.id)
                #expect(ModelLibraryFeature.reduce(&state, .openDemo(selection)).isEmpty)
                #expect(state.selection == selection)
                #expect(state.session == nil)
                #expect(state.access(to: selection) == .preview(.missing))
                let artifactID = section.model.defaultArtifactID
                #expect(ModelLibraryFeature.reduce(&state, .downloadRequested(artifactID)).isEmpty)
                #expect(state.installations[artifactID]?.failure == .downloadNotConfigured)
                #expect(ModelLibraryFeature.reduce(&state, .closeDemo).isEmpty)
                #expect(state.selection == nil)
            }
        }
    }

    @MainActor
    @Test("The browsing runner refuses direct download validation deletion and session release requests")
    func unavailableEffects() throws {
        let model = try #require(IOSDemoCatalog.make().models.first)
        let artifact = try #require(model.artifact(model.defaultArtifactID))
        let owner = UUID()
        let resource = ValidatedModelResource(
            modelID: model.id, artifactID: artifact.id, contract: artifact.contract, revision: "",
            upstreamRevision: nil, directory: URL(filePath: "/unavailable-model"), sizeBytes: 0)
        let request = LibraryTicket(owner: owner, number: 7)
        for (effect, expected) in [
            (ModelLibraryEffect.download(model, artifact, request), ModelLibraryFailure.downloadNotConfigured),
            (.validate(model, artifact, request, resource.directory), .invalidResource),
            (.delete(resource, request), .unavailable),
        ] {
            var calls = 0
            ModelLibraryBrowsingRunner.run(effect) { action in
                calls += 1
                if case .failed(let artifactID, let ticket, let failure) = action {
                    #expect(artifactID == artifact.id && ticket == request && failure == expected)
                } else {
                    Issue.record("A browsing runner must not report resource success")
                }
            }
            #expect(calls == 1)
        }
        let transfer = LibraryTicket(owner: owner, number: 8)
        ModelLibraryBrowsingRunner.run(.stopTransfer(artifactID: artifact.id, transfer)) { action in
            guard case .transferStopped(let id, let ticket) = action else {
                Issue.record("A browsing runner owns no pending transfer")
                return
            }
            #expect(id == artifact.id && ticket == transfer)
        }
        let session = ModelDemoSession(
            id: LibraryTicket(owner: owner, number: 9),
            selection: DemoSelection(modelID: model.id, demoID: "comparison"),
            resource: resource)
        let stop = LibraryTicket(owner: owner, number: 10)
        ModelLibraryBrowsingRunner.run(.stopDemo(session, stop)) { action in
            guard case .demoStopFailed(let id, let ticket, _) = action else {
                Issue.record("A browsing runner must not acknowledge releasing a resource it does not own")
                return
            }
            #expect(id == session.id && ticket == stop)
        }
    }
}
