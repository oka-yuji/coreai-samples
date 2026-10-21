import Foundation
import Testing

@testable import ModelLibraryFeature
@testable import ModelLibraryRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ModelResidencyTests {
    @Test("The Store preparation effect recovers from a permission change before it runs")
    func permissionChangeAndRetryUseTheStoreEffect() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let scene = harness.scene("A", capabilities: [.camera: .available])
        var events = harness.events.makeAsyncIterator()
        try await scene.download("image-fp16", &events)
        let library = scene.library
        library.send(.openDemo(.objectFinder))
        let session = try #require(library.state.session)
        _ = scene.rendering.content(for: session)
        library.send(.requestDemoPreparation(session.id))
        library.send(.capabilityChanged(.camera, .denied))
        await awaitDemoState(library) { $0.demoPreparation == .permissionRequired }
        #expect(await harness.leases(session) == 0)
        library.send(.capabilityChanged(.camera, .available))
        library.send(.requestDemoPreparation(session.id))
        await awaitDemoState(library) { $0.demoPreparation == .available }
        _ = scene.rendering.content(for: session)
        #expect(try scene.adapter("object-finder").start(.objectFinder, input: "after permission"))
        let stale = LibraryTicket(owner: scene.owner, number: session.id.number - 1)
        library.send(.demoPreparationChanged(stale, .occupied))
        #expect(library.state.demoPreparation == .available)
        library.send(.closeDemo)
        try await next(&events, "A stopping object-finder", "A stopDemo")
        #expect(await harness.leases(session) == 0)
    }

    @Test("A second owner of the same demo cannot bind or stop the resident owner's runtime")
    func rejectsAnotherOwnerOfTheSameSelection() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let first = harness.scene("A")
        let second = harness.scene("B")
        var events = harness.events.makeAsyncIterator()
        let adapter = try first.adapter("image-search")
        try await first.download("image-fp16", &events)
        second.library.send(.refresh)
        try await next(&events, "B inspect", "B inspect", "B inspect")
        first.library.send(.openDemo(.imageSearch))
        let owner = try await first.show()
        #expect(adapter.start(.imageSearch, input: "owner"))
        let originalGate = try #require(adapter.gate(for: .imageSearch))

        second.library.send(.openDemo(.imageSearch))
        let refused = try await second.show()
        #expect(refused.id.owner != owner.id.owner)
        #expect(adapter.contents == [owner])
        #expect(adapter.gate(for: .imageSearch)?.session == owner)
        #expect(originalGate.isPermitted())
        #expect(await harness.leases(refused) == 1)

        second.library.send(.closeDemo)
        try await next(&events, "B stopDemo")
        #expect(adapter.stops.isEmpty)
        #expect(adapter.entries[.imageSearch]?.runtime == owner.id)
        #expect(originalGate.isPermitted())
        #expect(await harness.leases(owner) == 1)
    }

    @Test("A resident model blocks another adapter until a successful stop and ignores an old stop after handoff")
    func waitsForSuccessfulReleaseBeforeHandoff() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let first = harness.scene("A")
        let second = harness.scene("B")
        var events = harness.events.makeAsyncIterator()
        try await first.download("text-fp16", &events)
        try await first.download("image-fp16", &events)
        second.library.send(.refresh)
        try await next(&events, "B inspect", "B inspect", "B inspect")
        let comparison = try first.adapter("comparison")
        let images = try second.adapter("image-search")
        first.library.send(.openDemo(.comparison))
        let owner = try await first.show()
        #expect(comparison.start(.comparison, input: "finished request with retained model"))
        second.library.send(.openDemo(.imageSearch))
        let candidate = try await second.show()
        #expect(second.library.state.demoPreparation == .occupied)
        #expect(images.contents.isEmpty && images.loads == 0)
        #expect(await harness.leases(candidate) == 0)

        comparison.holdsStop = true
        comparison.stopFailures = ["Still resident"]
        first.library.send(.closeDemo)
        try await next(&events, "A stopping comparison")
        _ = try await second.show()
        #expect(images.contents.isEmpty)
        comparison.releaseStops()
        try await next(&events, "A stopDemo")
        _ = try await second.show()
        #expect(second.library.state.demoPreparation == .occupied)
        #expect(await harness.leases(owner) == 1)

        comparison.holdsStop = false
        first.library.send(.retryDemoStop)
        try await next(&events, "A stopping comparison", "A stopDemo")
        #expect(await harness.leases(owner) == 0)
        #expect(try await second.show() == candidate)
        #expect(images.start(.imageSearch, input: "new owner"))
        let current = try #require(images.gate(for: .imageSearch))
        await harness.runner.run(.stopDemo(owner, LibraryTicket(owner: first.owner, number: 999))) { _ in }
        #expect(current.isPermitted() && images.stops.isEmpty)
        #expect(await harness.leases(candidate) == 1)
        #expect(comparison.stops == [owner, owner])
    }

    @Test("Preparation reserves residency before its lease suspends and an abandoned owner does not retain its store")
    func reservesBeforeSuspensionAndReleasesAbandonedPreparation() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        var first: LibraryScene? = harness.scene("A")
        let second = harness.scene("B")
        var events = harness.events.makeAsyncIterator()
        try await first?.download("text-fp16", &events)
        try await first?.download("image-fp16", &events)
        second.library.send(.refresh)
        try await next(&events, "B inspect", "B inspect", "B inspect")
        harness.ledger.hold()
        first?.library.send(.deleteRequested("image-fp16"))
        try await next(&events, "ledger")
        first?.library.send(.openDemo(.comparison))
        let owner = try #require(first?.library.state.session)
        _ = first?.rendering.content(for: owner)
        let preparing = Task { await harness.runner.prepare(owner) }
        if let library = first?.library { await awaitDemoState(library) { $0.demoPreparation == .preparing } }

        second.library.send(.openDemo(.comparison))
        let candidate = try await second.show()
        #expect(second.library.state.demoPreparation == .occupied)
        #expect(try second.adapter("comparison").contents.isEmpty)
        weak var library = first?.library
        harness.runner.abandon(owner: owner.id.owner)
        first = nil
        #expect(library == nil)
        harness.ledger.release()
        await preparing.value
        await harness.runner.release(owner: owner.id.owner)
        #expect(await harness.leases(owner) == 0)
        #expect(try await second.show() == candidate)
        #expect(try second.adapter("comparison").start(.comparison, input: "after cleanup"))
        #expect(await harness.leases(candidate) == 1)
    }

    @Test("A preparation whose owner disappears without closing gives up its residency and lease")
    func releasesPreparationOfADisappearedOwner() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        var first: LibraryScene? = harness.scene("A")
        let second = harness.scene("B")
        var events = harness.events.makeAsyncIterator()
        try await first?.download("text-fp16", &events)
        try await first?.download("image-fp16", &events)
        second.library.send(.refresh)
        try await next(&events, "B inspect", "B inspect", "B inspect")
        harness.ledger.hold()
        first?.library.send(.deleteRequested("image-fp16"))
        try await next(&events, "ledger")
        first?.library.send(.openDemo(.comparison))
        let owner = try #require(first?.library.state.session)
        _ = first?.rendering.content(for: owner)
        let preparing = Task { await harness.runner.prepare(owner) }
        if let library = first?.library { await awaitDemoState(library) { $0.demoPreparation == .preparing } }

        weak var library = first?.library
        first = nil
        #expect(library == nil)
        harness.ledger.release()
        await preparing.value
        await harness.runner.reattempt()
        #expect(await harness.leases(owner) == 0)
        second.library.send(.openDemo(.comparison))
        let candidate = try await second.show()
        #expect(second.library.state.demoPreparation == .available)
        #expect(try second.adapter("comparison").start(.comparison, input: "after an owner disappeared"))
        #expect(await harness.leases(candidate) == 1)
    }
}
