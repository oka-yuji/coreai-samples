import FeatureStore
import Foundation
import ModelLibraryFeature
import Testing

struct ModelLibraryTests {
    typealias Operation = ModelLibraryFeature.Operation

    private let owner = UUID()
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
            id: "artifact", title: "FP32", contract: "contract-v1", platforms: [.iOS], sizeBytes: 100,
            download: download,
            files: [ModelArtifactFile(path: "model.bin", bytes: 100, sha256: String(repeating: "b", count: 64))],
            visibility: artifactVisibility)
        let model = LibraryModel(
            id: "model", name: "Example Model", category: "Information Extraction", searchTerms: ["entity"],
            artifacts: [artifact], defaultArtifactID: "artifact", platforms: [.iOS], visibility: modelVisibility,
            demos: demos.map { ModelDemoLink(demoID: $0.id, platforms: [.iOS], visibility: linkVisibility) })
        return try ModelCatalog(models: [model], demos: demos)
    }

    private func resource(artifactID: String = "artifact") -> ValidatedModelResource {
        ValidatedModelResource(
            modelID: "model", artifactID: artifactID, contract: "contract-v1",
            revision: String(repeating: "a", count: 40), upstreamRevision: nil,
            directory: URL(filePath: "/test/model"), sizeBytes: 100)
    }

    private func ticket(_ number: Int) -> LibraryTicket { LibraryTicket(owner: owner, number: number) }

    private func libraryState(_ catalog: ModelCatalog) -> ModelLibraryState {
        ModelLibraryState(
            catalog: catalog, environment: LibraryEnvironment(platform: .iOS, connectedDemos: ["extract", "compare"]),
            owner: owner)
    }

    private func reduce(_ state: inout ModelLibraryState, _ action: ModelLibraryAction) -> [Operation] {
        ModelLibraryFeature.reduce(&state, action)
    }

    private func installedState() throws -> ModelLibraryState {
        var state = libraryState(try catalog())
        _ = reduce(&state, .refresh)
        let request = try #require(state.installations["artifact"]?.requestID)
        _ = reduce(&state, .inspected("artifact", request, resource()))
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
        var state = libraryState(try catalog())
        #expect(reduce(&state, .openDemo(second)).isEmpty)
        #expect(state.selection == second)
        #expect(state.session == nil)
        #expect(state.access(to: second) == .preview(.missing))
        #expect(!state.permits(ticket(0), selection: second))
        let unavailable = DemoSelection(modelID: "model", demoID: "unknown")
        #expect(reduce(&state, .openDemo(unavailable)).isEmpty)
        #expect(state.selection == second)
        #expect(state.access(to: unavailable) == .unavailable)
    }

    @Test("Validation grants the selected demo without issuing an inference effect")
    func downloadValidation() throws {
        var state = libraryState(try catalog())
        let model = try #require(state.catalog.models.first)
        let artifact = try #require(model.artifacts.first)
        _ = reduce(&state, .openDemo(second))
        #expect(
            reduce(&state, .downloadRequested("artifact"))
                == [.run(.download(model, artifact, ticket(1)), id: .resource("artifact"))])
        #expect(reduce(&state, .downloadRequested("artifact")).isEmpty)
        #expect(state.session == nil)
        let directory = resource().directory
        #expect(
            reduce(&state, .downloaded("artifact", ticket(1), directory))
                == [.run(.validate(model, artifact, ticket(1), directory), id: .resource("artifact"))])
        #expect(state.session == nil)
        #expect(reduce(&state, .validated("artifact", ticket(1), resource())).isEmpty)
        let session = try #require(state.session)
        #expect(state.selection == second)
        #expect(state.permits(session.id, selection: second))
        #expect(!state.permits(session.id, selection: first))
        #expect(state.installations["artifact"]?.phase == .ready)
    }

    @Test("Cancellation waits for acknowledgement and discards old completion", arguments: [false, true])
    func cancellationBarrier(validating: Bool) throws {
        var state = libraryState(try catalog())
        _ = reduce(&state, .openDemo(first))
        _ = reduce(&state, .downloadRequested("artifact"))
        if validating { _ = reduce(&state, .downloaded("artifact", ticket(1), resource().directory)) }
        #expect(
            reduce(&state, .cancelRequested("artifact"))
                == [
                    .cancel(.resource("artifact")),
                    .run(.stopTransfer(artifactID: "artifact", ticket(1)), id: .stopping("artifact")),
                ])
        #expect(state.installations["artifact"]?.phase == .stopping)
        #expect(reduce(&state, .downloadRequested("artifact")).isEmpty)
        #expect(reduce(&state, .validated("artifact", ticket(1), resource())).isEmpty)
        #expect(reduce(&state, .failed("artifact", ticket(1), .operation("late"))).isEmpty)
        #expect(state.installations["artifact"]?.phase == .stopping)
        _ = reduce(&state, .transferStopped("artifact", ticket(1)))
        _ = reduce(&state, .downloadRequested("artifact"))
        let snapshot = state
        _ = reduce(&state, .downloaded("artifact", ticket(1), resource().directory))
        _ = reduce(&state, .transferStopped("artifact", ticket(1)))
        _ = reduce(&state, .failed("artifact", ticket(1), .operation("late")))
        #expect(state == snapshot)
        #expect(state.selection == first)
    }

    @Test("Progress rejects invalid values, stale requests and backwards updates")
    func progress() throws {
        #expect(ModelTransferProgress(completedBytes: -1, totalBytes: 100) == nil)
        #expect(ModelTransferProgress(completedBytes: 101, totalBytes: 100) == nil)
        #expect(ModelTransferProgress(completedBytes: 1, totalBytes: nil)?.fraction == nil)
        var state = libraryState(try catalog())
        _ = reduce(&state, .downloadRequested("artifact"))
        let progress = try #require(ModelTransferProgress(completedBytes: 60, totalBytes: 100))
        _ = reduce(&state, .progress("artifact", ticket(1), progress))
        #expect(state.installations["artifact"]?.phase == .downloading(progress))
        let snapshot = state
        let previous = try #require(ModelTransferProgress(completedBytes: 20, totalBytes: 100))
        _ = reduce(&state, .progress("artifact", ticket(1), previous))
        _ = reduce(&state, .progress("artifact", ticket(99), progress))
        #expect(state == snapshot)
        #expect(progress.fraction == 0.6)
    }

    @Test("Failed transfers can retry without changing the viewed demo")
    func failureAndRetry() throws {
        var state = libraryState(try catalog())
        _ = reduce(&state, .openDemo(second))
        _ = reduce(&state, .downloadRequested("artifact"))
        _ = reduce(&state, .failed("artifact", ticket(1), .operation("offline")))
        #expect(state.installations["artifact"]?.phase == .failed)
        #expect(state.selection == second)
        #expect(state.session == nil)
        #expect(!reduce(&state, .downloadRequested("artifact")).isEmpty)
        #expect(state.installations["artifact"]?.failure == nil)
        #expect(state.installations["artifact"]?.requestID == ticket(2))
    }

    @Test("Missing or unpinned distributions never start a download")
    func distributionValidation() throws {
        var missing = libraryState(try catalog(download: nil))
        #expect(reduce(&missing, .downloadRequested("artifact")).isEmpty)
        #expect(missing.installations["artifact"]?.failure == .downloadNotConfigured)
        let invalid = ModelDownload(repository: "owner/model", revision: "main")
        var state = libraryState(try catalog(download: invalid))
        #expect(reduce(&state, .downloadRequested("artifact")).isEmpty)
        #expect(state.installations["artifact"]?.failure == .invalidDownload)
        #expect(!ModelDownload(repository: "../model", revision: String(repeating: "a", count: 40)).isValid)
    }

    @Test("Inspecting existing files cannot bypass missing or invalid distribution metadata", arguments: [false, true])
    func inspectedDistributionValidation(invalid: Bool) throws {
        let download = invalid ? ModelDownload(repository: "owner/model", revision: "main") : nil
        var state = libraryState(try catalog(download: download))
        _ = reduce(&state, .openDemo(first))
        _ = reduce(&state, .refresh)
        let request = try #require(state.installations["artifact"]?.requestID)
        _ = reduce(&state, .inspected("artifact", request, resource()))
        #expect(state.session == nil)
        #expect(state.installations["artifact"]?.phase == .failed)
        #expect(state.installations["artifact"]?.resource == nil)
        #expect(state.installations["artifact"]?.failure == (invalid ? .invalidDownload : .downloadNotConfigured))
    }

    @Test("A mismatched validation result never grants execution")
    func mismatchedResource() throws {
        var state = libraryState(try catalog())
        _ = reduce(&state, .openDemo(first))
        _ = reduce(&state, .refresh)
        _ = reduce(&state, .inspected("artifact", ticket(1), resource(artifactID: "different")))
        #expect(state.installations["artifact"]?.phase == .failed)
        #expect(state.installations["artifact"]?.failure == .invalidResource)
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
        _ = reduce(&state, .demoStopped(old.id, ticket(request.number + 1)))
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
        #expect(reduce(&state, .deleteRequested("artifact")).isEmpty)
        #expect(state.failure == .modelInUse)
        let session = try #require(state.session)
        _ = reduce(&state, .closeDemo)
        let stopped = try #require(state.stopRequestID)
        _ = reduce(&state, .demoStopped(session.id, stopped))
        let commands = reduce(&state, .deleteRequested("artifact"))
        let deletion = try #require(state.installations["artifact"]?.requestID)
        #expect(commands == [.run(.delete(resource(), deletion), id: .resource("artifact"))])
        _ = reduce(&state, .deleted("artifact", deletion))
        #expect(state.installations["artifact"]?.phase == .missing)
        #expect(state.sections.first?.demos.count == 2)
        _ = reduce(&state, .openDemo(first))
        #expect(state.selection == first)
        #expect(state.session == nil)
    }

    @Test("Failed deletion requires another inspection before the resource can run")
    func failedDeletion() throws {
        var state = try installedState()
        _ = reduce(&state, .deleteRequested("artifact"))
        let request = try #require(state.installations["artifact"]?.requestID)
        _ = reduce(&state, .failed("artifact", request, .operation("Delete failed")))
        _ = reduce(&state, .openDemo(first))
        #expect(state.session == nil)
        #expect(state.installations["artifact"]?.phase == .failed)
        #expect(state.installations["artifact"]?.resource == nil)
    }
}
