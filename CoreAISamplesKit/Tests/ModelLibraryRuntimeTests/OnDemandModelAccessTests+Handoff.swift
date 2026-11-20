import Foundation
import ModelLibraryStorage
import Testing

@testable import ModelLibraryFeature
@testable import ModelLibraryRuntime

extension OnDemandModelAccessTests {
    @Test("A resident demo in the same window prepares automatically after its previous explicit request releases")
    func preparesAfterSameWindowRequest() async throws {
        let harness = try makeHarness()
        defer { harness.close() }
        var events = harness.events.makeAsyncIterator()
        let source = harness.scene("source")
        try await source.download("image-fp16", &events)
        let library = sessionless(source, in: harness)
        let original = try #require(harness.runner.access(for: selection, in: library))
        let release = try await original.acquire()
        let other = sessionless(source, in: harness)
        let otherAccess = try #require(harness.runner.access(for: selection, in: other))
        library.send(.openDemo(residentSelection))
        let session = try #require(library.state.session)
        _ = harness.runner.gate(for: session, in: library)
        let preparing = Task.immediate { await harness.runner.prepare(session) }
        #expect(library.state.demoPreparation == .preparing)
        await #expect(throws: ManagedModelStoreError.inUse) { try await otherAccess.acquire() }
        #expect(await harness.store.activeLeases(identity: original.resource.identity) == 1)

        await release()
        await preparing.value
        #expect(library.state.demoPreparation == .available)
        #expect(harness.runner.bind(session, in: library)?.isPermitted() == true)
        library.send(.closeDemo)
        await awaitDemoState(library) { $0.session == nil }
        #expect(await harness.store.activeLeases(identity: original.resource.identity) == 0)
    }

    @Test("An abandoned window stops waiting for ordinary preparation without returning an in-flight explicit lease")
    func abandonedPreparationCannotBind() async throws {
        let harness = try makeHarness()
        defer { harness.close() }
        var events = harness.events.makeAsyncIterator()
        let source = harness.scene("source")
        try await source.download("image-fp16", &events)
        let library = sessionless(source, in: harness)
        let access = try #require(harness.runner.access(for: selection, in: library))
        let release = try await access.acquire()
        library.send(.openDemo(residentSelection))
        let session = try #require(library.state.session)
        let gate = harness.runner.gate(for: session, in: library)
        let preparing = Task.immediate { await harness.runner.prepare(session) }
        #expect(library.state.demoPreparation == .preparing)
        harness.runner.abandon(owner: library.state.owner)
        await preparing.value
        #expect(!gate.isPermitted())
        #expect(await harness.store.activeLeases(identity: access.resource.identity) == 1)
        await release()
        #expect(await harness.store.activeLeases(identity: access.resource.identity) == 0)
    }

    @Test(
        "Only the current preview can acquire after an old explicit request outlives a cancelled ordinary preparation")
    func supersededRequestsCannotRestart() async throws {
        let harness = try makeHarness()
        defer { harness.close() }
        var events = harness.events.makeAsyncIterator()
        let source = harness.scene("source")
        try await source.download("image-fp16", &events)
        let library = sessionless(source, in: harness)
        let original = try #require(harness.runner.access(for: selection, in: library))
        let release = try await original.acquire()

        library.send(.openDemo(residentSelection))
        let session = try #require(library.state.session)
        _ = harness.runner.gate(for: session, in: library)
        let preparing = Task.immediate { await harness.runner.prepare(session) }
        #expect(library.state.demoPreparation == .preparing)
        library.send(.closeDemo)
        await awaitDemoState(library) { $0.session == nil }
        library.send(.openDemo(selection))
        let interim = try #require(harness.runner.access(for: selection, in: library))
        let oldWaiter = Task.immediate { try await interim.acquire() }
        library.send(.suspendDemo)
        let current = try #require(harness.runner.access(for: selection, in: library))
        let activeWaiter = Task.immediate { try await current.acquire() }
        await release()
        await preparing.value
        await #expect(throws: CancellationError.self) { try await oldWaiter.value }
        let returned = try await activeWaiter.value
        #expect(library.state.selection == selection && library.state.session == nil)
        #expect(!harness.runner.gate(for: session, in: library).isPermitted())
        #expect(await harness.store.activeLeases(identity: original.resource.identity) == 1)
        await returned()
        #expect(await harness.store.activeLeases(identity: original.resource.identity) == 0)
    }

    @Test("Cancellation removes only its own waiter while the previous model and surviving request remain protected")
    func cancelsOneWindowWaiter() async throws {
        let harness = try makeHarness()
        defer { harness.close() }
        var events = harness.events.makeAsyncIterator()
        let source = harness.scene("source")
        try await source.download("image-fp16", &events)
        let library = sessionless(source, in: harness)
        let original = try #require(harness.runner.access(for: selection, in: library))
        let release = try await original.acquire()
        library.send(.suspendDemo)
        let current = try #require(harness.runner.access(for: selection, in: library))
        let cancelled = Task.immediate { try await current.acquire() }
        let surviving = Task.immediate { try await current.acquire() }
        cancelled.cancel()
        await #expect(throws: CancellationError.self) { try await cancelled.value }
        #expect(await harness.store.activeLeases(identity: original.resource.identity) == 1)
        await release()
        let returned = try await surviving.value
        await returned()
        #expect(await harness.store.activeLeases(identity: original.resource.identity) == 0)
    }
}
