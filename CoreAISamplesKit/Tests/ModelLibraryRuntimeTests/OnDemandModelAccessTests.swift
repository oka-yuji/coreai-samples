import FeatureStore
import Foundation
import ModelLibraryStorage
import Testing

@testable import ModelLibraryFeature
@testable import ModelLibraryRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct OnDemandModelAccessTests {
    let selection = SpokenSearchCatalog.selection
    let residentSelection = DemoSelection(
        modelID: SpokenSearchCatalog.selection.modelID, demoID: "image-search")

    @Test("Rendering an installed demo without a resident session holds no lease until an explicit request")
    func previewAndIdempotentRelease() async throws {
        let harness = try makeHarness()
        defer { harness.close() }
        let adapter = FakeOnDemandAdapter()
        harness.registry.register(adapter)
        var events = harness.events.makeAsyncIterator()
        let source = harness.scene("source")
        _ = source.rendering.preview(for: selection)
        #expect(adapter.access == nil)
        try await source.download("image-fp16", &events)

        let library = sessionless(source, in: harness)
        let rendering = ModelLibraryRendering(registry: harness.registry, runner: harness.runner, library: library)
        _ = rendering.preview(for: selection)
        let access = try #require(adapter.access)
        #expect(access.isCurrent())
        let resource = access.resource
        #expect(library.state.session == nil)
        #expect(access.generation == library.state.previewGeneration)
        #expect(resource.registrationGeneration == (await harness.store.generation(of: resource.identity)))
        #expect(await harness.store.activeLeases(identity: resource.identity) == 0)
        #expect(adapter.ordinaryPreviews == 0)
        _ = rendering.preview(for: residentSelection)
        #expect(try source.adapter("image-search").previews == [residentSelection])

        let first = try await access.acquire()
        #expect(await harness.store.activeLeases(identity: resource.identity) == 1)
        await #expect(throws: ManagedModelStoreError.inUse) { try await access.acquire() }
        await #expect(throws: ManagedModelStoreError.inUse) {
            try await harness.store.delete(identity: resource.identity)
        }
        await first()
        let second = try await access.acquire()
        await first()
        #expect(await harness.store.activeLeases(identity: resource.identity) == 1)
        await second()
        #expect(await harness.store.activeLeases(identity: resource.identity) == 0)
    }

    @Test("A pending model request excludes other windows and gives back its reservation when the owner closes")
    func pendingAndAbandon() async throws {
        let harness = try makeHarness()
        defer { harness.close() }
        var events = harness.events.makeAsyncIterator()
        let source = harness.scene("source")
        try await source.download("image-fp16", &events)
        let first = sessionless(source, in: harness)
        let second = sessionless(source, in: harness)
        let firstAccess = try #require(harness.runner.access(for: selection, in: first))
        let secondAccess = try #require(harness.runner.access(for: selection, in: second))

        harness.ledger.hold()
        source.library.send(.downloadRequested("text-fp16"))
        try await next(&events, "source download", "ledger")
        let pending = Task.immediate { try await firstAccess.acquire() }
        let competing = Task.immediate {
            do {
                let release = try await secondAccess.acquire()
                await release()
                return false
            } catch {
                return error as? ManagedModelStoreError == .inUse
            }
        }
        source.library.send(.openDemo(residentSelection))
        let ordinary = try #require(source.library.state.session)
        _ = harness.runner.gate(for: ordinary, in: source.library)
        let preparation = Task.immediate { await harness.runner.prepare(ordinary) }
        #expect(source.library.state.demoPreparation == .occupied)
        harness.runner.abandon(owner: first.state.owner)
        harness.ledger.release()
        await preparation.value
        #expect(await competing.value)
        await #expect(throws: CancellationError.self) { try await pending.value }
        #expect(await harness.store.activeLeases(identity: firstAccess.resource.identity) == 0)
        let release = try await secondAccess.acquire()
        await release()
    }

    @Test("Cancelling a request during its ledger check releases its reservation without acquiring a model")
    func cancelledAcquisition() async throws {
        let harness = try makeHarness()
        defer { harness.close() }
        var events = harness.events.makeAsyncIterator()
        let source = harness.scene("source")
        try await source.download("image-fp16", &events)
        let preview = sessionless(source, in: harness)
        let access = try #require(harness.runner.access(for: selection, in: preview))
        harness.ledger.hold()
        source.library.send(.downloadRequested("text-fp16"))
        try await next(&events, "source download", "ledger")
        let pending = Task.immediate { try await access.acquire() }
        pending.cancel()
        harness.ledger.release()
        await #expect(throws: CancellationError.self) { try await pending.value }
        #expect(await harness.store.activeLeases(identity: access.resource.identity) == 0)
        let release = try await access.acquire()
        await release()
    }

    @Test("Resident and explicit requests exclude one another through abandonment until their work releases the model")
    func residentAndHeldLease() async throws {
        let harness = try makeHarness()
        defer { harness.close() }
        var events = harness.events.makeAsyncIterator()
        let resident = harness.scene("resident")
        try await resident.download("image-fp16", &events)
        let preview = sessionless(resident, in: harness)
        let access = try #require(harness.runner.access(for: selection, in: preview))
        let identity = access.resource.identity

        harness.runner.register(resident.library)
        let generation = resident.library.state.previewGeneration
        resident.library.send(.openDemo(residentSelection))
        #expect(resident.library.state.previewGeneration == generation)
        #expect(harness.runner.access(for: residentSelection, in: resident.library) == nil)
        _ = try await resident.show()
        await #expect(throws: ManagedModelStoreError.inUse) { try await access.acquire() }
        resident.library.send(.closeDemo)
        try await next(&events, "resident stopDemo")
        let release = try await access.acquire()
        resident.library.send(.openDemo(residentSelection))
        let blocked = try #require(resident.library.state.session)
        _ = harness.runner.gate(for: blocked, in: resident.library)
        await harness.runner.prepare(blocked)
        #expect(resident.library.state.demoPreparation == .occupied)
        preview.send(.closeDemo)
        await #expect(throws: CancellationError.self) { try await access.acquire() }
        #expect(await harness.store.activeLeases(identity: identity) == 1)
        harness.runner.abandon(owner: preview.state.owner)
        await harness.runner.release(owner: preview.state.owner)
        #expect(await harness.store.activeLeases(identity: identity) == 1)
        await #expect(throws: ManagedModelStoreError.inUse) { try await harness.store.delete(identity: identity) }
        await release()
        await harness.runner.prepare(blocked)
        #expect(resident.library.state.demoPreparation == .available)
        await release()
        #expect(await harness.store.activeLeases(identity: identity) == 1)
        resident.library.send(.closeDemo)
        try await next(&events, "resident stopDemo")
        #expect(await harness.store.activeLeases(identity: identity) == 0)
    }

    @Test(
        "A closed, suspended or reselected preview cannot regain permission when navigation returns to the same model")
    func selectionGenerations() async throws {
        let harness = try makeHarness()
        defer { harness.close() }
        var events = harness.events.makeAsyncIterator()
        let source = harness.scene("source")
        try await source.download("image-fp16", &events)
        let library = sessionless(source, in: harness)
        let original = try #require(harness.runner.access(for: selection, in: library))
        #expect(harness.runner.access(for: residentSelection, in: library) == nil)
        library.send(.queryChanged("text"))
        #expect(library.state.previewGeneration == original.generation)
        var progressState = library.state
        _ = ModelLibraryFeature.reduce(&progressState, .downloadRequested("text-int4"))
        let transfer = try #require(progressState.installations["text-int4"]?.requestID)
        let progress = try #require(ModelTransferProgress(completedBytes: 1, totalBytes: 2))
        _ = ModelLibraryFeature.reduce(&progressState, .progress("text-int4", transfer, progress))
        #expect(progressState.previewGeneration == original.generation)
        library.send(.suspendDemo)
        await #expect(throws: CancellationError.self) { try await original.acquire() }
        let suspended = try #require(harness.runner.access(for: selection, in: library))
        library.send(.selectArtifact(modelID: selection.modelID, artifactID: "image-int4"))
        #expect(harness.runner.access(for: selection, in: library) == nil)
        library.send(.selectArtifact(modelID: selection.modelID, artifactID: "image-fp16"))
        await #expect(throws: CancellationError.self) { try await suspended.acquire() }
        let previous = try #require(harness.runner.access(for: selection, in: library))
        library.send(.closeDemo)
        await awaitDemoState(library) { $0.selection == nil }
        library.send(.openDemo(selection))
        let current = try #require(harness.runner.access(for: selection, in: library))
        #expect(current.generation > previous.generation)
        #expect(current.isCurrent() && !previous.isCurrent())
        await #expect(throws: CancellationError.self) { try await previous.acquire() }
        let release = try await current.acquire()
        await release()
        let replacement = Store<ModelLibraryFeature>(initialState: library.state, run: harness.runner.run)
        harness.runner.register(replacement)
        await #expect(throws: CancellationError.self) { try await current.acquire() }
        let replacementAccess = try #require(harness.runner.access(for: selection, in: replacement))
        let replacementRelease = try await replacementAccess.acquire()
        await replacementRelease()
    }

    @Test("An incorrect managed directory fails without leaking a lease or blocking another owner")
    func mismatchedDirectory() async throws {
        let harness = try makeHarness()
        defer { harness.close() }
        var events = harness.events.makeAsyncIterator()
        let source = harness.scene("source")
        try await source.download("image-fp16", &events)
        let resource = try #require(source.library.state.installations["image-fp16"]?.resource)
        var unstamped = resource
        unstamped.registrationGeneration = nil
        #expect(unstamped == resource)
        let legacy = sessionless(source, in: harness, resource: unstamped)
        #expect(harness.runner.access(for: selection, in: legacy) == nil)
        let misplaced = ValidatedModelResource(
            modelID: resource.modelID, artifactID: resource.artifactID, contract: resource.contract,
            revision: resource.revision, upstreamRevision: resource.upstreamRevision, directory: harness.base,
            sizeBytes: resource.sizeBytes, registrationGeneration: resource.registrationGeneration)
        let invalid = sessionless(source, in: harness, resource: misplaced)
        let badAccess = try #require(harness.runner.access(for: selection, in: invalid))
        await #expect(throws: ManagedModelStoreError.notInstalled) { try await badAccess.acquire() }
        #expect(await harness.store.activeLeases(identity: resource.identity) == 0)
        let valid = sessionless(source, in: harness)
        let goodAccess = try #require(harness.runner.access(for: selection, in: valid))
        let release = try await goodAccess.acquire()
        await release()
    }

    @Test("An installation replaced under the same identity is rejected by its old preview and a new preview works")
    func replacedRegistration() async throws {
        let harness = try makeHarness()
        defer { harness.close() }
        var events = harness.events.makeAsyncIterator()
        let first = harness.scene("first")
        try await first.download("image-fp16", &events)
        let stale = sessionless(first, in: harness)
        let oldAccess = try #require(harness.runner.access(for: selection, in: stale))
        let identity = oldAccess.resource.identity
        _ = try await harness.store.delete(identity: identity)
        await #expect(throws: ManagedModelStoreError.notInstalled) { try await oldAccess.acquire() }
        await awaitDemoState(stale) { $0.installations[identity.artifactID]?.phase == .missing }
        let second = harness.scene("second")
        try await second.download("image-fp16", &events)
        let current = sessionless(second, in: harness)
        let newAccess = try #require(harness.runner.access(for: selection, in: current))
        #expect(oldAccess.resource == newAccess.resource)
        #expect(oldAccess.resource.registrationGeneration != newAccess.resource.registrationGeneration)
        await #expect(throws: CancellationError.self) { try await oldAccess.acquire() }
        #expect(await harness.store.activeLeases(identity: identity) == 0)
        let release = try await newAccess.acquire()
        await release()
    }

    func makeHarness() throws -> RuntimeHarness {
        let descriptor = try #require(IOSDemoCatalog.make().demo(selection.demoID))
        return try RuntimeHarness(
            embeddingModelID: selection.modelID, additionalEmbeddingArtifacts: ["image-int4"],
            demoOverrides: [selection.demoID: descriptor])
    }

    func sessionless(
        _ source: LibraryScene, in harness: RuntimeHarness, resource: ValidatedModelResource? = nil
    ) -> Store<ModelLibraryFeature> {
        var state = ModelLibraryState(
            catalog: harness.catalog, environment: source.library.state.environment, owner: UUID())
        state.installations = source.library.state.installations
        if let resource { state.installations[resource.artifactID]?.resource = resource }
        let library = Store<ModelLibraryFeature>(initialState: state, run: harness.runner.run)
        harness.runner.register(library)
        library.send(.openDemo(selection))
        #expect(library.state.selection == selection && library.state.session == nil)
        return library
    }
}
