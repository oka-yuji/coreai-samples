import FeatureStore
import Foundation
import ModelLibraryStorage
import SwiftUI
import Testing

@testable import ModelLibraryFeature
@testable import ModelLibraryRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ModelLibraryLifecycleTests {
    @Test(
        "SESSION-05 Redraws, EN and JP, temporary inactivity, permission waits, background and scene end keep the selection, and returning alone never resumes execution"
    )
    func lifecycle() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let scene = harness.scene("A", capabilities: [.camera: .undetermined])
        var events = harness.events.makeAsyncIterator()
        let library = scene.library
        let adapter = try scene.adapter("object-finder")
        library.send(.openDemo(DemoSelection.objectFinder))
        try await scene.download("image-fp16", &events)
        let first = try await scene.show()
        let gate = try #require(adapter.gate(for: DemoSelection.objectFinder))
        #expect(gate.isPermitted())
        let ready = library.state
        _ = scene.rendering.content(for: first)
        #expect(ModelLibraryRendering.actions(entering: .inactive, capabilities: [.camera: .denied]).isEmpty)
        #expect(library.state == ready && adapter.contents == [first, first] && adapter.executions.isEmpty)

        let denied = ModelLibraryRendering.actions(
            entering: .active, capabilities: [.camera: .denied, .network: .available])
        #expect(denied.count == 2)
        denied.forEach(library.send)
        #expect(!gate.isPermitted() && !adapter.start(DemoSelection.objectFinder, input: "denied"))
        ModelLibraryRendering.actions(entering: .active, capabilities: [.camera: .available]).forEach(library.send)
        #expect(library.state.session == first && gate.isPermitted() && adapter.executions.isEmpty)
        #expect(adapter.start(DemoSelection.objectFinder, input: "cup"))

        let background = ModelLibraryRendering.actions(entering: .background, capabilities: [.camera: .available])
        background.forEach(library.send)
        try await next(&events, "A stopping object-finder", "A stopDemo")
        let resumed = try #require(library.state.session)
        #expect(library.state.selection == DemoSelection.objectFinder && resumed.id != first.id && !gate.isPermitted())
        ModelLibraryRendering.actions(entering: .active, capabilities: [.camera: .available]).forEach(library.send)
        #expect(library.state.session == resumed && adapter.executions == [first] && adapter.stops == [first])
        #expect(
            adapter.entries[DemoSelection.objectFinder]?.runtime == nil
                && adapter.entries[DemoSelection.objectFinder]?.input == "cup")

        library.send(.suspendDemo)
        try await next(&events, "A stopping object-finder", "A stopDemo")
        #expect(library.state.selection == DemoSelection.objectFinder && adapter.stops == [first, resumed])
        #expect(adapter.executions == [first] && adapter.loads == 1)
    }

    @Test(
        "SESSION-05 Conflicting owners keep their own sessions and leases until each owner has stopped its demo"
    )
    func owners() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let first = harness.scene("A")
        let second = harness.scene("B")
        var events = harness.events.makeAsyncIterator()
        try await first.download("image-fp16", &events)
        second.library.send(.refresh)
        try await next(&events, "B inspect", "B inspect", "B inspect")
        first.library.send(.openDemo(DemoSelection.imageSearch))
        let owned = try await first.show()
        second.library.send(.deleteRequested("image-fp16"))
        try await next(&events, "B delete")
        #expect(second.library.state.installations["image-fp16"]?.failure == .modelInUse)
        #expect(await harness.store.generation(of: owned.identity) != nil)

        second.library.send(.refresh)
        try await next(&events, "B inspect", "B inspect", "B inspect")
        second.library.send(.openDemo(DemoSelection.imageSearch))
        let shared = try await second.show()
        #expect(await harness.leases(owned) == 2)
        first.library.send(.closeDemo)
        try await next(&events, "A stopping image-search", "A stopDemo")
        #expect(await harness.leases(owned) == 1)
        #expect(try second.adapter("image-search").start(DemoSelection.imageSearch, input: "B"))
        second.library.send(.closeDemo)
        try await next(&events, "B stopping image-search", "B stopDemo")
        second.library.send(.deleteRequested("image-fp16"))
        try await next(&events, "B delete")
        #expect(second.library.state.installations["image-fp16"] == ModelInstallation())
        #expect(await harness.leases(shared) == 0)
        #expect(await harness.store.generation(of: shared.identity) == nil)
    }

    @Test(
        "SESSION-02 A resource deleted by another owner before the lease stops the demo and returns it to the download state on the same screen"
    )
    func lostResource() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let first = harness.scene("A")
        let second = harness.scene("B")
        var events = harness.events.makeAsyncIterator()
        let adapter = try first.adapter("image-search")
        try await first.download("image-fp16", &events)
        second.library.send(.refresh)
        try await next(&events, "B inspect", "B inspect", "B inspect")
        second.library.send(.deleteRequested("image-fp16"))
        try await next(&events, "B delete")
        first.library.send(.openDemo(DemoSelection.imageSearch))
        let stale = try await first.show()
        try await next(&events, "A stopping image-search", "A stopDemo")
        let library = first.library.state
        #expect(library.session == nil && library.selection == DemoSelection.imageSearch)
        #expect(library.installations["image-fp16"] == ModelInstallation())
        #expect(library.access(to: DemoSelection.imageSearch) == .preview(.missing))
        #expect(!adapter.start(DemoSelection.imageSearch, input: "deleted") && adapter.stops == [stale])
        #expect(await harness.leases(stale) == 0)

        try await first.download("image-fp16", &events)
        let restored = try await first.show()
        #expect(restored.id != stale.id && restored.identity == stale.identity)
        #expect(adapter.start(DemoSelection.imageSearch, input: "restored"))
        #expect(adapter.executions == [restored])
        #expect(await harness.leases(restored) == 1)
    }

    @Test("SESSION-02 An explicit start is rejected until the session's lease has been verified")
    func startWaitsForLease() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let scene = harness.scene("A")
        var events = harness.events.makeAsyncIterator()
        let adapter = try scene.adapter("comparison")
        try await scene.download("text-fp16", &events)
        harness.ledger.hold()
        scene.library.send(.downloadRequested("image-fp16"))
        try await next(&events, "A download", "ledger")
        scene.library.send(.openDemo(DemoSelection.comparison))
        let session = try #require(scene.library.state.session)
        _ = scene.rendering.content(for: session)
        scene.runner.provide(session)
        await Task.yield()
        #expect(!adapter.start(DemoSelection.comparison, input: "before the lease"))
        harness.ledger.release()
        try await next(&events, "A validate")
        await scene.runner.prepare(session)
        #expect(adapter.start(DemoSelection.comparison, input: "after the lease"))
        #expect(adapter.executions == [session] && adapter.rejections == 1)
    }

    @Test("SESSION-05 A scene whose library store is released still stops its demo once and releases the lease")
    func abandonedScene() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        var scene: LibraryScene? = harness.scene("A")
        var events = harness.events.makeAsyncIterator()
        let adapter = try #require(harness.adapters["image-search"])
        try await scene?.download("image-fp16", &events)
        scene?.library.send(.openDemo(DemoSelection.imageSearch))
        let session = try #require(try await scene?.show())
        #expect(adapter.start(DemoSelection.imageSearch, input: "kept"))
        #expect(await harness.leases(session) == 1)
        adapter.holdsStop = true
        weak let library = scene?.library
        scene?.library.send(.suspendDemo)
        harness.runner.abandon(owner: session.id.owner)
        scene = nil
        #expect(library == nil)
        try await next(&events, "A stopping image-search")
        adapter.releaseStops()
        await harness.runner.release(owner: session.id.owner)
        #expect(adapter.stops == [session] && adapter.entries[DemoSelection.imageSearch]?.runtime == nil)
        #expect(adapter.entries[DemoSelection.imageSearch]?.input == "kept")
        #expect(await harness.leases(session) == 0)
    }

    @Test(
        "SESSION-06 Five registered fake adapters share the host and the operation gate, and an unregistered demo stays browse-only"
    )
    func sharedHost() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let scene = harness.scene("A")
        var events = harness.events.makeAsyncIterator()
        let library = scene.library
        #expect(scene.registry.connectedDemos == Set(registeredDemos))
        #expect(library.state.environment.connectedDemos == Set(registeredDemos))
        try await scene.download("text-fp16", &events)
        try await scene.download("image-fp16", &events)
        for selection in DemoSelection.registered {
            let adapter = try scene.adapter(selection.demoID)
            _ = scene.rendering.preview(for: selection)
            library.send(.openDemo(selection))
            let session = try await scene.show()
            #expect(session.selection == selection && adapter.contents == [session])
            #expect(adapter.previews == [selection] && adapter.start(selection, input: selection.demoID))
            let gate = try #require(adapter.gate(for: selection))
            library.send(.closeDemo)
            try await next(&events, "A stopping " + selection.demoID, "A stopDemo")
            #expect(adapter.stops == [session] && !gate.isPermitted() && !adapter.start(selection, input: "late"))
            #expect(await harness.leases(session) == 0)
        }
        library.send(.openDemo(DemoSelection.notes))
        #expect(library.state.access(to: DemoSelection.notes) == .preview(.ready) && library.state.session == nil)
        _ = scene.rendering.preview(for: DemoSelection.notes)
        #expect(scene.adapters.values.allSatisfy { $0.previews.count == 1 && $0.executions.count == 1 })
        let orphan = ModelDemoSession(
            id: LibraryTicket(owner: scene.owner, number: 999), selection: DemoSelection.notes,
            resource: try #require(library.state.installations["text-fp16"]?.resource))
        var acknowledged: [ModelLibraryAction] = []
        await scene.runner.run(.stopDemo(orphan, LibraryTicket(owner: scene.owner, number: 1000))) {
            acknowledged.append($0)
        }
        guard case .demoStopped(orphan.id, _) = try #require(acknowledged.first) else {
            Issue.record("A demo without an adapter has nothing to release")
            return
        }
    }
}
