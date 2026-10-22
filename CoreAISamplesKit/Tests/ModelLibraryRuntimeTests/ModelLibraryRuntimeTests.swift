import FeatureStore
import Foundation
import ModelLibraryStorage
import SwiftUI
import Testing

@testable import ModelLibraryFeature
@testable import ModelLibraryRuntime

@MainActor
final class Outcomes {
    private(set) var actions: [String] = []

    func record(_ action: ModelLibraryAction) {
        switch action {
        case .demoStopped(_, let ticket): actions.append("stopped \(ticket.number)")
        case .demoStopFailed(_, let ticket, let message): actions.append("failed \(ticket.number) \(message)")
        default: actions.append("unexpected")
        }
    }
}

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ModelLibraryRuntimeTests {

    @Test(
        "SESSION-01 Only an explicit operation on a validated artifact with a connected adapter starts work, and browsing, download completion, language changes and explanation toggles start nothing"
    )
    func explicitStart() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let scene = harness.scene("A")
        var events = harness.events.makeAsyncIterator()
        let library = scene.library
        library.send(.refresh)
        try await next(&events, "A inspect", "A inspect", "A inspect")
        for selection in DemoSelection.registered {
            library.send(.openDemo(selection))
            #expect(library.state.access(to: selection) == .preview(.missing))
            _ = scene.rendering.preview(for: selection)
            library.send(.closeDemo)
        }
        library.send(.openDemo(DemoSelection.comparison))
        try await scene.download("text-fp16", &events)
        let session = try #require(library.state.session)
        #expect(session.selection == DemoSelection.comparison && session.identity.artifactID == "text-fp16")
        let adapters = DemoSelection.registered.compactMap { scene.adapters[$0.demoID] }
        #expect(adapters.map(\.previews) == DemoSelection.registered.map { [$0] })
        #expect(adapters.allSatisfy { $0.contents.isEmpty && $0.stops.isEmpty && $0.executions.isEmpty })

        let adapter = try scene.adapter("comparison")
        try await scene.show()
        let shown = library.state
        for _ in 0..<2 {
            _ = scene.rendering.content(for: session)
            scene.runner.provide(session)
            await scene.runner.prepare(session)
        }
        #expect(library.state == shown && adapter.contents == [session, session, session])
        #expect(adapter.executions.isEmpty && adapter.loads == 0 && adapter.rejections == 0)
        #expect(await harness.leases(session) == 1)
        #expect(adapter.start(DemoSelection.comparison, input: "spec"))
        #expect(adapter.executions == [session] && adapter.loads == 1)
        #expect(adapters.filter { $0 !== adapter }.allSatisfy { $0.executions.isEmpty && $0.contents.isEmpty })
    }

    @Test(
        "SESSION-02 Direct operations on missing, stopping, deleted or other-variant artifacts and late callbacks are rejected, and the next demo never starts before the stop is acknowledged"
    )
    func rejectsStaleOperations() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let scene = harness.scene("A")
        var events = harness.events.makeAsyncIterator()
        let library = scene.library
        let compare = try scene.adapter("comparison")
        let extract = try scene.adapter("extraction")
        library.send(.openDemo(DemoSelection.comparison))
        let missing = ModelDemoSession(
            id: LibraryTicket(owner: scene.owner, number: 1), selection: DemoSelection.comparison,
            resource: ValidatedModelResource(
                modelID: "text", artifactID: "text-fp16", contract: "embeddinggemma2-fp16-safe-s128-v70",
                revision: distributionRevision, upstreamRevision: upstreamRevision,
                directory: harness.base.appending(path: "missing"), sizeBytes: 1))
        #expect(!scene.runner.gate(for: missing, in: library).isPermitted())

        try await scene.download("text-fp16", &events)
        let first = try await scene.show()
        #expect(compare.start(DemoSelection.comparison, input: "first"))
        compare.holdsStop = true
        library.send(.openDemo(DemoSelection.extraction))
        try await next(&events, "A stopping comparison")
        #expect(!compare.start(DemoSelection.comparison, input: "while stopping"))
        library.send(.openDemo(DemoSelection.inspection))
        library.send(.closeDemo)
        #expect(library.state.pendingDestination == .demo(DemoSelection.extraction) && library.state.session == first)
        #expect(extract.contents.isEmpty && library.state.access(to: DemoSelection.extraction) == .stopping)
        compare.releaseStops()
        try await next(&events, "A stopDemo")
        let second = try await scene.show()
        #expect(
            second.selection == DemoSelection.extraction && extract.contents == [second] && compare.stops == [first])
        #expect(!compare.start(DemoSelection.comparison, input: "late callback"))

        try await scene.download("text-int4", &events)
        let fp16Gate = try #require(extract.gate(for: DemoSelection.extraction))
        library.send(.selectArtifact(modelID: "text", artifactID: "text-int4"))
        try await next(&events, "A stopping extraction", "A stopDemo")
        let third = try await scene.show()
        #expect(third.identity.artifactID == "text-int4" && third.selection == DemoSelection.extraction)
        #expect(!fp16Gate.isPermitted() && extract.start(DemoSelection.extraction, input: "int4"))
        let int4Gate = try #require(extract.gate(for: DemoSelection.extraction))

        library.send(.closeDemo)
        try await next(&events, "A stopping extraction", "A stopDemo")
        _ = scene.rendering.content(for: third)
        scene.runner.provide(third)
        await scene.runner.prepare(third)
        #expect(await harness.leases(third) == 0)
        library.send(.deleteRequested("text-int4"))
        try await next(&events, "A delete")
        #expect(library.state.installations["text-int4"] == ModelInstallation())
        library.send(.openDemo(DemoSelection.extraction))
        #expect(
            library.state.access(to: DemoSelection.extraction) == .preview(.missing) && library.state.session == nil)
        #expect(!int4Gate.isPermitted() && !extract.start(DemoSelection.extraction, input: "deleted"))
        #expect(compare.rejections == 2 && extract.rejections == 1)
    }

    @Test(
        "SESSION-03 A failed stop is never reported as success, and retries, double closes and back or switch races resolve each request exactly once"
    )
    func resolvesStopsOnce() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let scene = harness.scene("A")
        var events = harness.events.makeAsyncIterator()
        let library = scene.library
        let adapter = try scene.adapter("comparison")
        library.send(.openDemo(DemoSelection.comparison))
        try await scene.download("text-fp16", &events)
        let session = try await scene.show()
        #expect(await harness.leases(session) == 1)

        adapter.holdsStop = true
        adapter.stopFailures = ["Still releasing"]
        library.send(.closeDemo)
        try await next(&events, "A stopping comparison")
        let stopping = library.state
        for action in [ModelLibraryAction.closeDemo, .suspendDemo, .openDemo(DemoSelection.extraction), .retryDemoStop]
        {
            library.send(action)
        }
        #expect(library.state == stopping)
        adapter.releaseStops()
        try await next(&events, "A stopDemo")
        #expect(library.state.failure == .operation("Still releasing") && library.state.session == session)
        #expect(library.state.pendingDestination == .library && adapter.stops == [session])
        #expect(await harness.leases(session) == 1)
        #expect(!adapter.start(DemoSelection.comparison, input: "blocked"))

        adapter.holdsStop = false
        library.send(.retryDemoStop)
        try await next(&events, "A stopping comparison", "A stopDemo")
        #expect(library.state.session == nil && library.state.selection == nil && library.state.failure == nil)
        #expect(adapter.stops == [session, session])
        #expect(await harness.leases(session) == 0)

        library.send(.openDemo(DemoSelection.comparison))
        let reopened = try await scene.show()
        adapter.holdsStop = true
        adapter.stopFailures = ["Busy"]
        let outcomes = Outcomes()
        let tickets = [901, 902].map { LibraryTicket(owner: scene.owner, number: $0) }
        let first = Task { await scene.runner.run(.stopDemo(reopened, tickets[0])) { outcomes.record($0) } }
        try await next(&events, "A stopping comparison")
        let second = Task { await scene.runner.run(.stopDemo(reopened, tickets[1])) { outcomes.record($0) } }
        await Task.yield()
        adapter.releaseStops()
        await first.value
        await second.value
        #expect(outcomes.actions.sorted() == ["failed 901 Busy", "failed 902 Busy"])
        #expect(adapter.stops == [session, session, reopened])
        #expect(await harness.leases(reopened) == 1)
        adapter.holdsStop = false
        await scene.runner.run(.stopDemo(reopened, tickets[1])) { outcomes.record($0) }
        #expect(outcomes.actions.last == "stopped 902")
        #expect(await harness.leases(reopened) == 0)
    }

    @Test(
        "SESSION-03 Going back while the lease is still being acquired releases it once before the stop is acknowledged"
    )
    func stopsDuringLeaseAcquisition() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let scene = harness.scene("A")
        var events = harness.events.makeAsyncIterator()
        let library = scene.library
        try await scene.download("text-fp16", &events)
        harness.ledger.hold()
        library.send(.downloadRequested("image-fp16"))
        try await next(&events, "A download", "ledger")
        library.send(.openDemo(DemoSelection.comparison))
        let session = try #require(library.state.session)
        _ = scene.rendering.content(for: session)
        scene.runner.provide(session)
        await awaitDemoState(library) { $0.demoPreparation == .preparing }
        library.send(.closeDemo)
        #expect(library.state.session == session)
        #expect(library.state.pendingDestination == .library)
        #expect(try scene.adapter("comparison").stops.isEmpty)
        harness.ledger.release()
        try await next(&events, "A stopDemo", "A validate")
        #expect(library.state.session == nil && library.state.selection == nil)
        #expect(await harness.leases(session) == 0)
        library.send(.deleteRequested("text-fp16"))
        try await next(&events, "A delete")
        #expect(library.state.installations["text-fp16"] == ModelInstallation())
    }

    @Test("SESSION-02 A late content appearance while a stopped session is still being released takes no lease")
    func ignoresProvisionDuringRelease() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let scene = harness.scene("A")
        var events = harness.events.makeAsyncIterator()
        let library = scene.library
        try await scene.download("text-fp16", &events)
        library.send(.openDemo(DemoSelection.comparison))
        let session = try await scene.show()
        harness.ledger.hold()
        library.send(.downloadRequested("image-fp16"))
        try await next(&events, "A download", "ledger")
        library.send(.closeDemo)
        try await next(&events, "A stopping comparison")
        await Task.yield()
        scene.runner.provide(session)
        await Task.yield()
        harness.ledger.release()
        try await next(&events, "A stopDemo", "A validate")
        await scene.runner.prepare(session)
        #expect(library.state.session == nil)
        #expect(await harness.leases(session) == 0)
    }

    @Test(
        "SESSION-04 Inputs, results and started saves of each model and demo survive the runtime release, and deletion or a variant change never reuses an old runtime"
    )
    func keepsDataAndReleasesRuntime() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let scene = harness.scene("A")
        var events = harness.events.makeAsyncIterator()
        let library = scene.library
        let adapter = try scene.adapter("comparison")
        library.send(.openDemo(DemoSelection.comparison))
        try await scene.download("text-fp16", &events)
        let first = try await scene.show()
        #expect(adapter.start(DemoSelection.comparison, input: "first"))
        let saving = Task { await adapter.save(DemoSelection.comparison) }
        try await next(&events, "A saving comparison")
        library.send(.closeDemo)
        try await next(&events, "A stopping comparison", "A stopDemo")
        #expect(library.state.session == nil && adapter.entries[DemoSelection.comparison]?.runtime == nil)
        #expect(adapter.saved.isEmpty)
        #expect(await harness.leases(first) == 0)
        adapter.completeSaves()
        await saving.value
        #expect(adapter.saved == ["first@text-fp16"])

        try await scene.download("text-int4", &events)
        library.send(.openDemo(DemoSelection.comparison))
        let second = try await scene.show()
        #expect(second.id != first.id && adapter.entries[DemoSelection.comparison]?.input == "first")
        #expect(adapter.start(DemoSelection.comparison, input: "second") && adapter.loads == 2)
        library.send(.selectArtifact(modelID: "text", artifactID: "text-int4"))
        try await next(&events, "A stopping comparison", "A stopDemo")
        let third = try await scene.show()
        #expect(third.identity.artifactID == "text-int4")
        #expect(await harness.leases(second) == 0)
        #expect(adapter.start(DemoSelection.comparison, input: "third") && adapter.loads == 3)
        #expect(adapter.entries[DemoSelection.comparison]?.runtime == third.id)

        library.send(.closeDemo)
        try await next(&events, "A stopping comparison", "A stopDemo")
        library.send(.deleteRequested("text-int4"))
        try await next(&events, "A delete")
        library.send(.openDemo(DemoSelection.comparison))
        #expect(library.state.session == nil)
        #expect(await harness.store.generation(of: third.identity) == nil)
        try await scene.download("text-int4", &events)
        let fourth = try await scene.show()
        #expect(fourth.identity == third.identity && fourth.id != third.id)
        #expect(adapter.start(DemoSelection.comparison, input: "fourth") && adapter.loads == 4)
        #expect(
            adapter.entries[DemoSelection.comparison]?.results == [
                "first@text-fp16", "second@text-fp16", "third@text-int4", "fourth@text-int4",
            ])
        #expect(adapter.saved == ["first@text-fp16"])
    }
}
