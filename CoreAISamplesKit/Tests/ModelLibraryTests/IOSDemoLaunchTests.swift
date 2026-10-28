import FeatureStore
import Foundation
import ModelLibraryUI
import Testing

@testable import ModelLibraryFeature

private struct Distribution: Decodable {
    struct Artifact: Decodable {
        let modelID: String
        let artifactID: String
        let repository: String
        let revision: String
        let root: String
        let upstream: ModelUpstream
        let files: [ModelArtifactFile]
        let sizeBytes: Int64
    }

    let artifacts: [Artifact]
    let totalBytes: Int64
}

struct IOSDemoLaunchTests {
    @Test(
        "Ordinary launch opens Models and explicit diagnostics preserve their destinations",
        arguments: [
            ([String](), IOSDemoLaunch.models),
            (["app", "--cpu"], .models),
            (["app", "--speech-reference-check", "test-run"], .speechReferenceDiagnostic("test-run")),
            (["app", "--speech-reference-check=test-run"], .speechReferenceDiagnostic("test-run")),
            (["app", "--speech-reference-check"], .models),
            (["app", "--extract-eval"], .models),
            (["app", "--extract-eval", "fixture"], .extractionEvaluation),
            (["app", "--extract-eval=fixture"], .extractionEvaluation),
            (["app", "--image-search-eval", "fixture"], .imageSearchEvaluation),
            (["app", "--image-search-eval", "--extract-eval", "fixture"], .imageSearchEvaluation),
            (["app", "--model-library-selftest"], .models),
            (["app", "--model-library-selftest", "report.json"], .modelLibrarySelfTest),
            (["app", "--model-library-selftest=report.json"], .modelLibrarySelfTest),
            (["app", "--extract-eval", "fixture", "--model-library-selftest", "report.json"], .extractionEvaluation),
            (["app", "--model-library-selftest", "report.json", "--legacy-demos"], .modelLibrarySelfTest),
            (["app", "--model-library-selftest", "report.json", "--acquire", "all"], .modelLibraryAcquisition("all")),
            (["app", "--model-library-selftest=report.json", "--acquire=fp16"], .modelLibraryAcquisition("fp16")),
            (["app", "--acquire", "all", "--model-library-selftest", "report.json"], .modelLibraryAcquisition("all")),
            (["app", "--model-library-selftest", "report.json", "--acquire"], .modelLibrarySelfTest),
            (["app", "--acquire", "all"], .models),
            (["app", "--model-library-selftest", "--acquire", "all"], .modelLibraryAcquisition("all")),
            (["app", "--extract-eval", "e", "--model-library-selftest", "r", "--acquire", "a"], .extractionEvaluation),
            (["app", "--image-search"], .legacyDemos),
            (["app", "--object-finder"], .legacyDemos),
            (["app", "--truck-inspection"], .legacyDemos),
            (["app", "--legacy-demos"], .legacyDemos),
        ])
    func destination(arguments: [String], expected: IOSDemoLaunch) {
        #expect(IOSDemoLaunch.resolve(arguments: arguments) == expected)
    }

    @Test(
        "ADOPT-01 The iOS catalog presents two models and six demos whose artifacts carry the published distribution definitions with every file, size, checksum and upstream"
    )
    func catalog() throws {
        let catalog = try IOSDemoCatalog.make()
        let sections = catalog.sections(platform: .iOS, audience: .release)
        #expect(sections.map(\.model.name) == ["GLiNER2.5-multi", "EmbeddingGemma 2"])
        #expect(
            sections.map { $0.demos.map(\.id) } == [
                ["comparison", "extraction", "inspection"], ["image-search", "video-search", "object-finder"],
            ])
        #expect(sections.map { $0.artifacts.map(\.title) } == [["FP32"], ["FP16", "INT4/INT8 PTQ"]])
        #expect(
            catalog.models.map(\.defaultArtifactID) == [
                "gliner2.5-multi-fp32-t256-t512", "embeddinggemma2-fp16-safe-s128-v70",
            ])
        #expect(catalog.sections(platform: .macOS, audience: .release).isEmpty)

        let url = try #require(
            Bundle.module.url(forResource: "distribution", withExtension: "json", subdirectory: "Resources"))
        let published = try JSONDecoder().decode(Distribution.self, from: Data(contentsOf: url))
        let artifacts = catalog.models.flatMap(\.artifacts)
        #expect(artifacts.map(\.id).sorted() == published.artifacts.map(\.artifactID).sorted())
        let contracts = [
            "gliner2.5-multi-fp32-t256-t512": "gliner-gl4-fp32-t256-t512",
            "embeddinggemma2-fp16-safe-s128-v70": "embeddinggemma2-fp16-safe-s128-v70",
            "embeddinggemma2-int4-int8-ptq-s128-v70": "embeddinggemma2-int4-int8-ptq-s128-v70",
        ]
        for definition in published.artifacts {
            let model = try #require(catalog.model(containing: definition.artifactID))
            let artifact = try #require(model.artifact(definition.artifactID))
            #expect(model.id == definition.modelID)
            #expect(artifact.contract == contracts[artifact.id])
            #expect(
                artifact.download
                    == ModelDownload(
                        repository: definition.repository, revision: definition.revision, root: definition.root))
            #expect(artifact.download?.isValid == true)
            #expect(artifact.upstream == definition.upstream)
            #expect(artifact.files.sorted { $0.path < $1.path } == definition.files.sorted { $0.path < $1.path })
            #expect(artifact.sizeBytes == definition.sizeBytes)
            #expect(definition.sizeBytes == definition.files.reduce(0) { $0 + $1.bytes })
            #expect(artifact.visibility == .published && artifact.platforms == [.iOS])
        }
        #expect(published.totalBytes == artifacts.reduce(0) { $0 + ($1.sizeBytes ?? 0) })
    }

    @Test("A translated demo search retains its model while filtering unrelated demos")
    func localizedSearch() throws {
        let catalog = try IOSDemoCatalog.make(searchTerms: ["comparison": ["comparar productos"]])
        let matches = catalog.sections(platform: .iOS, audience: .release, query: "comparar")
        #expect(matches.map(\.id) == ["gliner2.5-multi"])
        #expect(matches.first?.demos.map(\.id) == ["comparison"])
        #expect(catalog.sections(platform: .iOS, audience: .release, query: "embeddinggemma").first?.demos.count == 3)
        let video = try IOSDemoCatalog.make(searchTerms: ["video-search": ["doga no bamen"]])
        let found = video.sections(platform: .iOS, audience: .release, query: "doga no bamen")
        #expect(found.map(\.id) == ["embeddinggemma2"])
        #expect(found.first?.demos.map(\.id) == ["video-search"])
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
    @Test(
        "Browsing all production demos and requesting their downloads through the browsing runner never creates a session"
    )
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
            for (index, demo) in section.demos.enumerated() {
                let selection = DemoSelection(modelID: section.id, demoID: demo.id)
                #expect(ModelLibraryFeature.reduce(&state, .openDemo(selection)).isEmpty)
                #expect(state.selection == selection)
                #expect(state.session == nil)
                #expect(state.access(to: selection) == .preview(index == 0 ? .missing : .failed))
                let artifactID = section.model.defaultArtifactID
                let commands = ModelLibraryFeature.reduce(&state, .downloadRequested(artifactID))
                let artifact = try #require(section.model.artifact(artifactID))
                let request = try #require(state.installations[artifactID]?.requestID)
                #expect(commands == [.run(.download(section.model, artifact, request), id: .resource(artifactID))])
                for command in commands {
                    if case .run(let effect, _) = command {
                        ModelLibraryBrowsingRunner.run(effect) { action in
                            #expect(ModelLibraryFeature.reduce(&state, action).isEmpty)
                        }
                    }
                }
                #expect(state.installations[artifactID]?.failure == .downloadNotConfigured)
                #expect(state.installations[artifactID]?.phase == .failed && state.session == nil)
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
