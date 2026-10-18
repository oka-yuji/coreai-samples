import FeatureStore
import Foundation
import ModelLibraryFeature
import Testing

struct ModelLibraryTests {
    typealias Operation = ModelLibraryFeature.Operation

    private let first = DemoSelection(modelID: "model", demoID: "extract")
    private let second = DemoSelection(modelID: "model", demoID: "compare")

    private func catalog(
        modelVisibility: CatalogVisibility = .published,
        artifactVisibility: CatalogVisibility = .published,
        linkVisibility: CatalogVisibility = .published,
        download: ModelDownload? = ModelDownload(repository: "owner/model", revision: String(repeating: "a", count: 40))
    ) throws -> ModelCatalog {
        let demos = ["extract", "compare"].map { id in
            DemoDescriptor(
                id: id, title: id == "extract" ? "Extract demo" : "Compare demo",
                systemImage: "doc.text", summary: "Inspect a document", platforms: [.iOS, .macOS])
        }
        let artifact = ModelArtifact(
            id: "artifact", contract: "contract-v1", sizeBytes: 100,
            download: download, visibility: artifactVisibility)
        let model = LibraryModel(
            id: "model", name: "Example Model", category: "Information Extraction", searchTerms: ["entity"],
            artifact: artifact, platforms: [.iOS], visibility: modelVisibility,
            demos: demos.map { ModelDemoLink(demoID: $0.id, platforms: [.iOS], visibility: linkVisibility) })
        return try ModelCatalog(models: [model], demos: demos)
    }

    private func resource(artifactID: String = "artifact") -> ValidatedModelResource {
        ValidatedModelResource(
            modelID: "model", artifactID: artifactID, contract: "contract-v1",
            directory: URL(filePath: "/test/model"), sizeBytes: 100)
    }

    private func reduce(_ state: inout ModelLibraryState, _ action: ModelLibraryAction) -> [Operation] {
        ModelLibraryFeature.reduce(&state, action)
    }

    private func installedState() throws -> ModelLibraryState {
        var state = ModelLibraryState(catalog: try catalog(), platform: .iOS)
        _ = reduce(&state, .refresh)
        let request = try #require(state.installations["model"]?.requestID)
        _ = reduce(&state, .inspected("model", request, resource()))
        return state
    }

    @Test("Catalog visibility requires model, artifact and demo-link publication")
    func publicationIntersection() throws {
        let catalogs = [
            try catalog(modelVisibility: .hidden), try catalog(artifactVisibility: .hidden),
            try catalog(linkVisibility: .hidden), try catalog(modelVisibility: .development),
        ]
        for catalog in catalogs {
            #expect(catalog.sections(platform: .iOS, audience: .release).isEmpty)
        }
        let development = try catalog(modelVisibility: .development)
        #expect(development.sections(platform: .iOS, audience: .development).count == 1)
        #expect(try catalog().sections(platform: .macOS, audience: .release).isEmpty)
    }

    @Test("Search retains model ownership when a demo title matches")
    func search() throws {
        let catalog = try catalog()
        let found = catalog.sections(platform: .iOS, audience: .release, query: " COMPARE ")
        #expect(found.map(\.id) == ["model"])
        #expect(found.first?.demos.map(\.id) == ["compare"])
        #expect(catalog.sections(platform: .iOS, audience: .release, query: "entity").first?.demos.count == 2)
        #expect(catalog.sections(platform: .iOS, audience: .release, query: "missing").isEmpty)
    }

    @Test("Catalogs reject duplicate identities and unknown demo links")
    func catalogValidation() throws {
        let catalog = try catalog()
        let model = try #require(catalog.models.first)
        let demo = try #require(catalog.demos.first)
        #expect(throws: ModelCatalogError.duplicateModel("model")) {
            try ModelCatalog(models: [model, model], demos: catalog.demos)
        }
        #expect(throws: ModelCatalogError.duplicateDemo(demo.id)) {
            try ModelCatalog(models: [], demos: [demo, demo])
        }
        #expect(throws: ModelCatalogError.invalidLink("model", "extract")) {
            try ModelCatalog(models: [model], demos: [])
        }
    }

    @Test("Missing models allow browsing without granting an execution session")
    func previewOnly() throws {
        var state = ModelLibraryState(catalog: try catalog(), platform: .iOS)
        #expect(reduce(&state, .openDemo(second)).isEmpty)
        #expect(state.selection == second)
        #expect(state.session == nil)
        #expect(state.access(to: second) == .preview(.missing))
        #expect(!state.permits(0, selection: second))
        let unavailable = DemoSelection(modelID: "model", demoID: "unknown")
        #expect(reduce(&state, .openDemo(unavailable)).isEmpty)
        #expect(state.selection == second)
        #expect(state.access(to: unavailable) == .unavailable)
    }

    @Test("Validation grants the selected demo without issuing an inference effect")
    func downloadValidation() throws {
        var state = ModelLibraryState(catalog: try catalog(), platform: .iOS)
        let model = try #require(state.catalog.models.first)
        _ = reduce(&state, .openDemo(second))
        #expect(reduce(&state, .downloadRequested("model")) == [.run(.download(model, 1), id: .resource("model"))])
        #expect(reduce(&state, .downloadRequested("model")).isEmpty)
        #expect(state.session == nil)
        let directory = resource().directory
        #expect(
            reduce(&state, .downloaded("model", 1, directory))
                == [.run(.validate(model, 1, directory), id: .resource("model"))])
        #expect(state.session == nil)
        #expect(reduce(&state, .validated("model", 1, resource())).isEmpty)
        let session = try #require(state.session)
        #expect(state.selection == second)
        #expect(state.permits(session.id, selection: second))
        #expect(!state.permits(session.id, selection: first))
        #expect(state.installations["model"]?.phase == .ready)
    }

    @Test("Cancellation waits for acknowledgement and discards old completion", arguments: [false, true])
    func cancellationBarrier(validating: Bool) throws {
        var state = ModelLibraryState(catalog: try catalog(), platform: .iOS)
        _ = reduce(&state, .openDemo(first))
        _ = reduce(&state, .downloadRequested("model"))
        if validating { _ = reduce(&state, .downloaded("model", 1, resource().directory)) }
        #expect(
            reduce(&state, .cancelRequested("model"))
                == [.cancel(.resource("model")), .run(.stopTransfer("model", 1), id: .stopping("model"))])
        #expect(state.installations["model"]?.phase == .stopping)
        #expect(reduce(&state, .downloadRequested("model")).isEmpty)
        #expect(reduce(&state, .validated("model", 1, resource())).isEmpty)
        #expect(reduce(&state, .failed("model", 1, .operation("late"))).isEmpty)
        #expect(state.installations["model"]?.phase == .stopping)
        _ = reduce(&state, .transferStopped("model", 1))
        _ = reduce(&state, .downloadRequested("model"))
        let snapshot = state
        _ = reduce(&state, .downloaded("model", 1, resource().directory))
        _ = reduce(&state, .transferStopped("model", 1))
        _ = reduce(&state, .failed("model", 1, .operation("late")))
        #expect(state == snapshot)
        #expect(state.selection == first)
    }

    @Test("Progress rejects invalid values, stale requests and backwards updates")
    func progress() throws {
        #expect(ModelTransferProgress(completedBytes: -1, totalBytes: 100) == nil)
        #expect(ModelTransferProgress(completedBytes: 101, totalBytes: 100) == nil)
        #expect(ModelTransferProgress(completedBytes: 1, totalBytes: nil)?.fraction == nil)
        var state = ModelLibraryState(catalog: try catalog(), platform: .iOS)
        _ = reduce(&state, .downloadRequested("model"))
        let progress = try #require(ModelTransferProgress(completedBytes: 60, totalBytes: 100))
        _ = reduce(&state, .progress("model", 1, progress))
        #expect(state.installations["model"]?.phase == .downloading(progress))
        let snapshot = state
        let previous = try #require(ModelTransferProgress(completedBytes: 20, totalBytes: 100))
        _ = reduce(&state, .progress("model", 1, previous))
        _ = reduce(&state, .progress("model", 99, progress))
        #expect(state == snapshot)
        #expect(progress.fraction == 0.6)
    }

    @Test("Failed transfers can retry without changing the viewed demo")
    func failureAndRetry() throws {
        var state = ModelLibraryState(catalog: try catalog(), platform: .iOS)
        _ = reduce(&state, .openDemo(second))
        _ = reduce(&state, .downloadRequested("model"))
        _ = reduce(&state, .failed("model", 1, .operation("offline")))
        #expect(state.installations["model"]?.phase == .failed)
        #expect(state.selection == second)
        #expect(state.session == nil)
        #expect(!reduce(&state, .downloadRequested("model")).isEmpty)
        #expect(state.installations["model"]?.failure == nil)
        #expect(state.installations["model"]?.requestID == 2)
    }

    @Test("Missing or unpinned distributions never start a download")
    func distributionValidation() throws {
        var missing = ModelLibraryState(catalog: try catalog(download: nil), platform: .iOS)
        #expect(reduce(&missing, .downloadRequested("model")).isEmpty)
        #expect(missing.installations["model"]?.failure == .downloadNotConfigured)
        let invalid = ModelDownload(repository: "owner/model", revision: "main")
        var state = ModelLibraryState(catalog: try catalog(download: invalid), platform: .iOS)
        #expect(reduce(&state, .downloadRequested("model")).isEmpty)
        #expect(state.installations["model"]?.failure == .invalidDownload)
        #expect(!ModelDownload(repository: "../model", revision: String(repeating: "a", count: 40)).isValid)
    }

    @Test("Inspecting existing files cannot bypass missing or invalid distribution metadata", arguments: [false, true])
    func inspectedDistributionValidation(invalid: Bool) throws {
        let download = invalid ? ModelDownload(repository: "owner/model", revision: "main") : nil
        var state = ModelLibraryState(catalog: try catalog(download: download), platform: .iOS)
        _ = reduce(&state, .openDemo(first))
        _ = reduce(&state, .refresh)
        let request = try #require(state.installations["model"]?.requestID)
        _ = reduce(&state, .inspected("model", request, resource()))
        #expect(state.session == nil)
        #expect(state.installations["model"]?.phase == .failed)
        #expect(state.installations["model"]?.resource == nil)
        #expect(state.installations["model"]?.failure == (invalid ? .invalidDownload : .downloadNotConfigured))
    }

    @Test("A mismatched validation result never grants execution")
    func mismatchedResource() throws {
        var state = ModelLibraryState(catalog: try catalog(), platform: .iOS)
        _ = reduce(&state, .openDemo(first))
        _ = reduce(&state, .refresh)
        _ = reduce(&state, .inspected("model", 1, resource(artifactID: "different")))
        #expect(state.installations["model"]?.phase == .failed)
        #expect(state.installations["model"]?.failure == .invalidResource)
        #expect(state.session == nil)
    }

    @Test("Demo switches wait for the old session to stop and reject stale acknowledgements")
    func demoSwitch() throws {
        var state = try installedState()
        _ = reduce(&state, .openDemo(first))
        let old = try #require(state.session)
        let commands = reduce(&state, .openDemo(second))
        let request = try #require(state.stopRequestID)
        #expect(commands == [.run(.stopDemo(old, request), id: .demo)])
        #expect(state.selection == first)
        #expect(!state.permits(old.id, selection: first))
        let snapshot = state
        _ = reduce(&state, .demoStopped(old.id, request + 1))
        _ = reduce(&state, .closeDemo)
        #expect(state == snapshot)
        _ = reduce(&state, .demoStopped(old.id, request))
        let new = try #require(state.session)
        #expect(state.selection == second)
        #expect(new.id != old.id)
        #expect(state.permits(new.id, selection: second))
        #expect(!state.permits(old.id, selection: first))
    }

    @Test("A failed stop keeps execution blocked until a matching retry succeeds")
    func stopFailure() throws {
        var state = try installedState()
        _ = reduce(&state, .openDemo(first))
        let session = try #require(state.session)
        _ = reduce(&state, .closeDemo)
        let firstRequest = try #require(state.stopRequestID)
        _ = reduce(&state, .demoStopFailed(session.id, firstRequest, "Still busy"))
        #expect(state.pendingDestination == .library)
        #expect(!state.permits(session.id, selection: first))
        _ = reduce(&state, .retryDemoStop)
        let retry = try #require(state.stopRequestID)
        #expect(retry != firstRequest)
        _ = reduce(&state, .demoStopped(session.id, firstRequest))
        #expect(state.session != nil)
        _ = reduce(&state, .demoStopped(session.id, retry))
        #expect(state.selection == nil)
        #expect(state.session == nil)
    }

    @Test("Deleting a model requires a released session and retains its demo definitions")
    func deletion() throws {
        var state = try installedState()
        _ = reduce(&state, .openDemo(first))
        #expect(reduce(&state, .deleteRequested("model")).isEmpty)
        #expect(state.failure == .modelInUse)
        let session = try #require(state.session)
        _ = reduce(&state, .closeDemo)
        let stopped = try #require(state.stopRequestID)
        _ = reduce(&state, .demoStopped(session.id, stopped))
        let commands = reduce(&state, .deleteRequested("model"))
        let deletion = try #require(state.installations["model"]?.requestID)
        #expect(commands == [.run(.delete(resource(), deletion), id: .resource("model"))])
        _ = reduce(&state, .deleted("model", deletion))
        #expect(state.installations["model"]?.phase == .missing)
        #expect(state.sections.first?.demos.count == 2)
        _ = reduce(&state, .openDemo(first))
        #expect(state.selection == first)
        #expect(state.session == nil)
    }

    @Test("Failed deletion requires another inspection before the resource can run")
    func failedDeletion() throws {
        var state = try installedState()
        _ = reduce(&state, .deleteRequested("model"))
        let request = try #require(state.installations["model"]?.requestID)
        _ = reduce(&state, .failed("model", request, .operation("Delete failed")))
        _ = reduce(&state, .openDemo(first))
        #expect(state.session == nil)
        #expect(state.installations["model"]?.phase == .failed)
        #expect(state.installations["model"]?.resource == nil)
    }
}
