import FeatureStore
import Foundation
import Testing

@testable import ModelLibraryFeature

private let distribution = LibraryFixture.distributionRevision
private let upstream = LibraryFixture.upstreamRevision

struct ModelLibraryContractTests {
    typealias Operation = ModelLibraryFeature.Operation

    private let fixture = LibraryFixture()
    private let camera = DemoSelection(modelID: "model", demoID: "camera")
    private let text = DemoSelection(modelID: "model", demoID: "text")

    private func reduce(_ state: inout ModelLibraryState, _ action: ModelLibraryAction) -> [Operation] {
        ModelLibraryFeature.reduce(&state, action)
    }

    @Test(
        "CONTRACT-01 Variants of one model keep independent installations, their model and demo membership, and the selected artifact"
    )
    func variantInstallations() throws {
        var state = fixture.state(try fixture.variantCatalog(), audience: .development)
        #expect(state.selectedArtifacts == ["model": "fp32"])
        #expect(state.artifacts(for: "model").map(\.id) == ["fp32", "int4"])
        #expect(Set(state.installations.keys) == ["fp32", "int4", "mac"])
        #expect(state.catalog.model(containing: "int4")?.id == "model")
        _ = reduce(&state, .openDemo(text))
        try fixture.install(&state, "fp32")
        let fp32 = try #require(state.session)
        #expect(fp32.selection == text && fp32.identity.artifactID == "fp32")
        #expect(state.installations["int4"] == ModelInstallation())

        let switching = reduce(&state, .selectArtifact(modelID: "model", artifactID: "int4"))
        let request = try #require(state.stopRequestID)
        #expect(switching == [.run(.stopDemo(fp32, request), id: .demo)])
        #expect(state.selectedArtifacts["model"] == "fp32" && state.pendingArtifact == "int4")
        #expect(reduce(&state, .demoStopped(fp32.id, request)).isEmpty)
        #expect(state.selectedArtifacts["model"] == "int4" && state.selection == text && state.session == nil)
        #expect(state.access(to: text) == .preview(.missing))
        #expect(state.installations["fp32"]?.phase == .ready)

        try fixture.install(&state, "int4")
        let int4 = try #require(state.session)
        #expect(int4.identity.artifactID == "int4" && int4.selection == text)
        _ = reduce(&state, .selectArtifact(modelID: "model", artifactID: "fp32"))
        let release = try #require(state.stopRequestID)
        _ = reduce(&state, .demoStopped(int4.id, release))
        #expect(state.session?.resource == fp32.resource)
        #expect(state.installations["int4"]?.phase == .ready)
        #expect(state.sections.map(\.id) == ["model"])
        #expect(state.sections.first?.demos.map(\.id) == ["camera", "text"])
    }

    @Test("CONTRACT-02 Direct requests for unpublished variants or unsupported platforms issue no side effects")
    func hiddenRequests() throws {
        let catalog = try fixture.variantCatalog()
        let model = try #require(catalog.model("model"))
        var release = fixture.state(catalog)
        #expect(release.artifacts(for: "model").map(\.id) == ["fp32"])
        for hidden in ["int4", "mac"] {
            #expect(reduce(&release, .downloadRequested(hidden)).isEmpty)
            #expect(reduce(&release, .deleteRequested(hidden)).isEmpty)
            #expect(reduce(&release, .selectArtifact(modelID: "model", artifactID: hidden)).isEmpty)
            #expect(release.failure == .unavailable)
            #expect(release.installations[hidden] == ModelInstallation())
        }
        #expect(release.selectedArtifacts == ["model": "fp32"])
        let fp32 = try #require(model.artifact("fp32"))
        #expect(
            reduce(&release, .refresh)
                == [.run(.inspect(model, fp32, LibraryTicket(owner: fixture.owner, number: 1)), id: .resource("fp32"))])

        var mac = fixture.state(catalog, platform: .macOS)
        #expect(mac.sections.isEmpty)
        #expect(reduce(&mac, .refresh).isEmpty)
        #expect(reduce(&mac, .openDemo(text)).isEmpty)
        #expect(mac.failure == .unavailable && mac.selection == nil)
        #expect(reduce(&mac, .downloadRequested("fp32")).isEmpty)
        #expect(mac.access(to: text) == .unavailable)
        #expect(mac.installations["fp32"] == ModelInstallation())
    }

    @Test(
        "CONTRACT-02 Required capabilities gate execution while optional capabilities never do",
        arguments: [
            (CapabilityState?.none, true), (.available, true), (.undetermined, true), (.denied, false),
            (.unsupported, false),
        ])
    func capabilityGate(cameraState: CapabilityState?, permitted: Bool) throws {
        var capabilities: [DemoCapability: CapabilityState] = [.microphone: .denied]
        capabilities[.camera] = cameraState
        var state = fixture.state(try fixture.variantCatalog(), capabilities: capabilities)
        _ = reduce(&state, .openDemo(camera))
        try fixture.install(&state, "fp32")
        let session = try #require(state.session)
        #expect(state.permits(session.id, selection: camera) == permitted)
        #expect(reduce(&state, .capabilityChanged(.camera, .available)).isEmpty)
        #expect(state.permits(session.id, selection: camera))
        #expect(reduce(&state, .capabilityChanged(.camera, .denied)).isEmpty)
        #expect(!state.permits(session.id, selection: camera))
        #expect(state.sections.first?.demos.map(\.id) == ["camera", "text"])
    }

    @Test("CONTRACT-02 A demo without a connected adapter stays browse-only after its artifact is installed")
    func unconnectedAdapter() throws {
        var state = fixture.state(try fixture.variantCatalog(), connected: ["text"])
        _ = reduce(&state, .openDemo(camera))
        try fixture.install(&state, "fp32")
        #expect(state.session == nil)
        #expect(!state.isConnected(camera) && state.isConnected(text))
        #expect(state.access(to: camera) == .preview(.ready))
        #expect(!state.permits(LibraryTicket(owner: fixture.owner, number: 1), selection: camera))
        #expect(reduce(&state, .openDemo(text)).isEmpty)
        let session = try #require(state.session)
        #expect(state.permits(session.id, selection: text))
    }

    @Test("CONTRACT-02 Forged tickets and mismatched artifact identities never pass the execution gate")
    func identityGate() throws {
        var state = fixture.state(try fixture.variantCatalog(), audience: .development)
        _ = reduce(&state, .openDemo(text))
        try fixture.install(&state, "fp32")
        let session = try #require(state.session)
        #expect(state.permits(session.id, selection: text))
        #expect(!state.permits(LibraryTicket(owner: UUID(), number: session.id.number), selection: text))
        #expect(!state.permits(LibraryTicket(owner: fixture.owner, number: session.id.number + 1), selection: text))
        #expect(!state.permits(session.id, selection: camera))
        #expect(reduce(&state, .selectArtifact(modelID: "other", artifactID: "fp32")).isEmpty)
        #expect(reduce(&state, .downloadRequested("model")).isEmpty)
        #expect(reduce(&state, .deleteRequested("unknown")).isEmpty)
        #expect(state.session == session && state.selectedArtifacts == ["model": "fp32"])
        state.selectedArtifacts["model"] = "int4"
        #expect(!state.permits(session.id, selection: text))
    }

    @Test(
        "CONTRACT-03 Distribution and upstream revisions are matched separately",
        arguments: [
            (distribution, upstream, true), (upstream, upstream, false),
            (distribution, distribution, false), (distribution, nil, false),
            (distribution, String(repeating: "f", count: 40), false),
        ])
    func revisionMatching(revision: String, upstream: String?, accepted: Bool) throws {
        var state = fixture.state(try fixture.variantCatalog())
        _ = reduce(&state, .openDemo(text))
        _ = reduce(&state, .refresh)
        let request = try #require(state.installations["fp32"]?.requestID)
        let found = fixture.resource("fp32", revision: revision, upstream: upstream)
        #expect(reduce(&state, .inspected("fp32", request, found)).isEmpty)
        #expect(state.installations["fp32"]?.phase == (accepted ? .ready : .failed))
        #expect(state.installations["fp32"]?.failure == (accepted ? nil : .invalidResource))
        #expect(state.session?.identity == (accepted ? found.identity : nil))
        #expect(found.identity.revision == revision && found.identity.contract == "fixture-fp32")
    }

    @Test("CONTRACT-03 Completions from other artifacts, requests, owners or old sessions are not adopted")
    func staleCompletions() throws {
        var state = fixture.state(try fixture.variantCatalog(), audience: .development)
        _ = reduce(&state, .openDemo(text))
        _ = reduce(&state, .downloadRequested("fp32"))
        let request = try #require(state.installations["fp32"]?.requestID)
        let foreign = LibraryTicket(owner: UUID(), number: request.number)
        let progress = try #require(ModelTransferProgress(completedBytes: 1, totalBytes: 4))
        let snapshot = state
        for action in [
            ModelLibraryAction.downloaded("int4", request, URL(filePath: "/fixture/int4")),
            .downloaded("fp32", foreign, URL(filePath: "/fixture/fp32")),
            .downloaded(
                "fp32", LibraryTicket(owner: fixture.owner, number: request.number + 1), URL(filePath: "/fixture")),
            .progress("fp32", foreign, progress), .failed("fp32", foreign, .operation("late")),
            .transferStopped("fp32", foreign), .validated("fp32", request, fixture.resource("fp32")),
        ] {
            #expect(reduce(&state, action).isEmpty)
        }
        #expect(state == snapshot)
        _ = reduce(&state, .downloaded("fp32", request, URL(filePath: "/fixture/fp32")))
        #expect(reduce(&state, .validated("fp32", foreign, fixture.resource("fp32"))).isEmpty)
        #expect(state.installations["fp32"]?.phase == .validating)
        _ = reduce(&state, .validated("fp32", request, fixture.resource("int4")))
        #expect(state.installations["fp32"]?.failure == .invalidResource && state.session == nil)

        try fixture.install(&state, "fp32")
        let old = try #require(state.session)
        _ = reduce(&state, .closeDemo)
        let stop = try #require(state.stopRequestID)
        #expect(reduce(&state, .demoStopped(LibraryTicket(owner: UUID(), number: old.id.number), stop)).isEmpty)
        #expect(reduce(&state, .demoStopped(old.id, LibraryTicket(owner: UUID(), number: stop.number))).isEmpty)
        #expect(state.session == old)
        _ = reduce(&state, .demoStopped(old.id, stop))
        _ = reduce(&state, .openDemo(text))
        let current = try #require(state.session)
        let reopened = state
        #expect(reduce(&state, .demoStopped(old.id, stop)).isEmpty)
        #expect(reduce(&state, .demoStopFailed(old.id, stop, "late")).isEmpty)
        #expect(state == reopened)
        #expect(current.id != old.id && !state.permits(old.id, selection: text))
    }

    @Test("CONTRACT-04 Download completion and variant changes never start inference")
    func noAutomaticInference() throws {
        var state = fixture.state(try fixture.variantCatalog(), audience: .development)
        _ = reduce(&state, .refresh)
        for artifactID in ["fp32", "int4"] {
            let request = try #require(state.installations[artifactID]?.requestID)
            #expect(reduce(&state, .inspected(artifactID, request, nil)).isEmpty)
        }
        #expect(reduce(&state, .selectArtifact(modelID: "model", artifactID: "int4")).isEmpty)
        #expect(reduce(&state, .selectArtifact(modelID: "model", artifactID: "fp32")).isEmpty)
        _ = reduce(&state, .openDemo(text))
        try fixture.install(&state, "fp32")
        let session = try #require(state.session)
        let switching = reduce(&state, .selectArtifact(modelID: "model", artifactID: "int4"))
        let request = try #require(state.stopRequestID)
        #expect(switching == [.run(.stopDemo(session, request), id: .demo)])
        #expect(reduce(&state, .demoStopped(session.id, request)).isEmpty)
        #expect(state.session == nil)
        try fixture.install(&state, "int4")
        #expect(state.session?.identity.artifactID == "int4")
    }

    @Test(
        "CONTRACT-04 A failed stop rejects the next execution and a retry keeps the session identity and selected artifacts"
    )
    func failedStopKeepsAttribution() throws {
        var state = fixture.state(try fixture.variantCatalog(), audience: .development)
        _ = reduce(&state, .openDemo(text))
        try fixture.install(&state, "fp32")
        let session = try #require(state.session)
        _ = reduce(&state, .selectArtifact(modelID: "model", artifactID: "int4"))
        let first = try #require(state.stopRequestID)
        _ = reduce(&state, .demoStopFailed(session.id, first, "Still busy"))
        #expect(!state.permits(session.id, selection: text) && state.access(to: text) == .stopping)
        for action in [
            ModelLibraryAction.openDemo(camera), .closeDemo, .selectArtifact(modelID: "model", artifactID: "fp32"),
            .deleteRequested("fp32"), .demoStopped(session.id, first),
        ] {
            #expect(reduce(&state, action).isEmpty)
        }
        #expect(state.failure == .operation("Still busy"))
        #expect(state.session?.identity == session.identity && state.selection == text)
        #expect(state.selectedArtifacts == ["model": "fp32"] && state.pendingArtifact == "int4")
        let retry = reduce(&state, .retryDemoStop)
        let second = try #require(state.stopRequestID)
        #expect(retry == [.run(.stopDemo(session, second), id: .demo)] && second != first)
        #expect(state.session?.identity == session.identity && state.selectedArtifacts == ["model": "fp32"])
        #expect(reduce(&state, .demoStopped(session.id, second)).isEmpty)
        #expect(state.selectedArtifacts == ["model": "int4"] && state.selection == text)
        #expect(state.session == nil && state.failure == nil)
    }

    @Test(
        "CONTRACT-05 Shared-demo and ten-demo fixtures keep definitions, search and explanations through the same registration",
        arguments: [
            [(model: "first", demos: ["shared", "solo"]), (model: "second", demos: ["shared"])],
            [(model: "wide", demos: (1...10).map { "demo-\($0)" })],
        ])
    func registrationFixtures(entries: [(model: String, demos: [String])]) throws {
        let catalog = try fixture.registeredCatalog(entries)
        var state = fixture.state(catalog)
        #expect(state.sections.map(\.id) == entries.map(\.model))
        #expect(state.sections.map { $0.demos.map(\.id) } == entries.map(\.demos))
        for entry in entries {
            for demoID in entry.demos {
                let selection = DemoSelection(modelID: entry.model, demoID: demoID)
                let found = catalog.sections(platform: .iOS, audience: .release, query: demoID + " demo")
                #expect(found.contains { $0.id == entry.model && $0.demos.map(\.id) == [demoID] })
                #expect(catalog.demo(demoID)?.explanation.first?.text == "Fixture explanation for " + demoID)
                #expect(reduce(&state, .openDemo(selection)).isEmpty)
                #expect(state.access(to: selection) == .preview(.missing))
                #expect(reduce(&state, .closeDemo).isEmpty)
            }
        }
        let lead = try #require(entries.first)
        let selection = DemoSelection(modelID: lead.model, demoID: try #require(lead.demos.first))
        _ = reduce(&state, .openDemo(selection))
        try fixture.install(&state, lead.model + "-fp32", modelID: lead.model)
        let session = try #require(state.session)
        #expect(state.permits(session.id, selection: selection))
        for other in entries.dropFirst() where other.demos.contains(selection.demoID) {
            let shared = DemoSelection(modelID: other.model, demoID: selection.demoID)
            #expect(state.access(to: shared) == .preview(.missing))
            #expect(!state.permits(session.id, selection: shared))
            #expect(state.installations[other.model + "-fp32"] == ModelInstallation())
        }
    }
}
