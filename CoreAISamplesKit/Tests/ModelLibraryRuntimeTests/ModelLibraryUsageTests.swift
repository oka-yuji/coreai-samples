import FeatureStore
import Foundation
import ModelLibraryStorage
import Testing

@testable import ModelLibraryFeature
@testable import ModelLibraryRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ModelLibraryUsageTests {
    private let identity = ModelArtifactIdentity(
        modelID: "text", artifactID: "text-fp16", contract: "embeddinggemma2-fp16-safe-s128-v70",
        revision: distributionRevision)

    private func scene(_ label: String, in harness: RuntimeHarness) -> LibraryScene {
        let scene = harness.scene(label)
        harness.runner.register(scene.library)
        return scene
    }

    private func leases(_ harness: RuntimeHarness) async -> Int {
        await harness.store.activeLeases(identity: identity)
    }

    @Test(
        "USE-01 A usage lease is granted only for an installed artifact that the owner's library has verified, and a refusal leaves no lease"
    )
    func grantsOnlyVerifiedArtifacts() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let first = scene("A", in: harness)
        let second = scene("B", in: harness)
        var events = harness.events.makeAsyncIterator()
        let runner = harness.runner

        #expect(await runner.acquireUsage(identity: identity, owner: first.owner) == nil)
        #expect(await leases(harness) == 0)
        try await first.download("text-fp16", &events)
        let resource = try #require(first.library.state.installations["text-fp16"]?.resource)

        let otherRevision = ModelArtifactIdentity(
            modelID: "text", artifactID: "text-fp16", contract: identity.contract,
            revision: String(repeating: "f", count: 40))
        #expect(await runner.acquireUsage(identity: otherRevision, owner: first.owner) == nil)
        #expect(await runner.acquireUsage(identity: identity, owner: second.owner) == nil)
        #expect(await runner.acquireUsage(identity: identity, owner: UUID()) == nil)
        #expect(await harness.store.activeLeases(identity: otherRevision) == 0)
        #expect(await leases(harness) == 0)

        #expect(await runner.acquireUsage(identity: identity, owner: first.owner) == resource)
        #expect(await leases(harness) == 1)
        await runner.releaseUsage(identity: identity, owner: first.owner)
        #expect(await leases(harness) == 0)

        _ = try await harness.store.delete(identity: identity)
        #expect(first.library.state.installations["text-fp16"]?.phase == .ready)
        #expect(await runner.acquireUsage(identity: identity, owner: first.owner) == nil)
        #expect(await leases(harness) == 0)

        first.library.send(.useRequested("text"))
        try await next(&events, "A acquireUsage")
        #expect(first.library.state.usages["text"]?.isActive == false)
        #expect(first.library.state.usages["text"]?.resource == nil)
        #expect(await leases(harness) == 0)
        first.library.send(.dismissFailure)
        #expect(first.library.state.usages["text"]?.isActive == false)
    }

    @Test(
        "USE-01 A use requested while another owner's deletion is being written is refused once the deletion lands and leaves no lease"
    )
    func refusesDuringConcurrentDeletion() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let first = scene("A", in: harness)
        let second = scene("B", in: harness)
        var events = harness.events.makeAsyncIterator()
        try await first.download("text-fp16", &events)
        second.library.send(.refresh)
        try await next(&events, "B inspect", "B inspect", "B inspect")
        #expect(second.library.state.installations["text-fp16"]?.phase == .ready)

        harness.ledger.hold()
        first.library.send(.deleteRequested("text-fp16"))
        try await next(&events, "ledger")
        second.library.send(.useRequested("text"))
        #expect(second.library.state.usages["text"]?.isActive == true)
        harness.ledger.release()
        try await next(&events, "A delete", "B acquireUsage")
        #expect(first.library.state.installations["text-fp16"] == ModelInstallation())
        #expect(second.library.state.usages["text"]?.resource == nil)
        #expect(second.library.state.usages["text"]?.isActive == false)
        #expect(await leases(harness) == 0)
    }

    @Test(
        "USE-02 Deleting a model in use is refused and keeps its installed state and location, and it is deleted once the usage is released"
    )
    func refusesDeletionWhileInUse() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let first = scene("A", in: harness)
        var events = harness.events.makeAsyncIterator()
        let library = first.library
        try await first.download("text-fp16", &events)
        let resource = try #require(library.state.installations["text-fp16"]?.resource)

        library.send(.useRequested("text"))
        try await next(&events, "A acquireUsage")
        #expect(library.state.usages["text"]?.resource == resource)
        #expect(await leases(harness) == 1)

        library.send(.deleteRequested("text-fp16"))
        try await next(&events, "A delete")
        let installation = try #require(library.state.installations["text-fp16"])
        #expect(installation.phase == .ready && installation.resource == resource)
        #expect(installation.failure == .modelInUse)
        #expect(await harness.store.generation(of: identity) != nil)
        #expect(FileManager.default.fileExists(atPath: resource.directory.appending(path: "tokenizer.json").path))

        library.send(.useEnded("text"))
        try await next(&events, "A releaseUsage")
        #expect(library.state.usages["text"] == nil)
        #expect(await leases(harness) == 0)
        library.send(.deleteRequested("text-fp16"))
        try await next(&events, "A delete")
        #expect(library.state.installations["text-fp16"] == ModelInstallation())
        #expect(await harness.store.generation(of: identity) == nil)
        #expect(!FileManager.default.fileExists(atPath: resource.directory.path))
    }

    @Test(
        "USE-03 Every usage lease of an owner returns with release(owner:), and two usages of one owner and model need two releases"
    )
    func returnsEveryLeaseOfAnOwner() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let first = scene("A", in: harness)
        let second = scene("B", in: harness)
        var events = harness.events.makeAsyncIterator()
        let runner = harness.runner
        try await first.download("text-fp16", &events)
        second.library.send(.refresh)
        try await next(&events, "B inspect", "B inspect", "B inspect")

        #expect(await runner.acquireUsage(identity: identity, owner: first.owner) != nil)
        #expect(await runner.acquireUsage(identity: identity, owner: first.owner) != nil)
        await runner.releaseUsage(identity: identity, owner: first.owner)
        #expect(await leases(harness) == 1)
        await #expect(throws: ManagedModelStoreError.inUse) { try await harness.store.delete(identity: identity) }

        second.library.send(.useRequested("text"))
        second.library.send(.useRequested("text"))
        try await next(&events, "B acquireUsage", "B acquireUsage")
        #expect(second.library.state.usages["text"]?.isActive == true)
        #expect(await leases(harness) == 3)
        await runner.release(owner: second.owner)
        #expect(await leases(harness) == 1)
        #expect(await runner.acquireUsage(identity: identity, owner: second.owner) == nil)
        await #expect(throws: ManagedModelStoreError.inUse) { try await harness.store.delete(identity: identity) }

        await runner.releaseUsage(identity: identity, owner: first.owner)
        await runner.releaseUsage(identity: identity, owner: first.owner)
        #expect(await leases(harness) == 0)
        first.library.send(.deleteRequested("text-fp16"))
        try await next(&events, "A delete")
        #expect(first.library.state.installations["text-fp16"] == ModelInstallation())
    }

    @Test("USE-03 A use that ends while its lease is still being taken returns that lease when it arrives")
    func endsDuringAcquisition() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        let first = scene("A", in: harness)
        var events = harness.events.makeAsyncIterator()
        let library = first.library
        try await first.download("text-fp16", &events)
        harness.ledger.hold()
        library.send(.downloadRequested("image-fp16"))
        try await next(&events, "A download", "ledger")

        library.send(.useRequested("text"))
        #expect(library.state.usages["text"]?.isActive == true)
        library.send(.useEnded("text"))
        #expect(library.state.usages["text"]?.isActive == true)
        harness.ledger.release()
        try await next(&events, "A validate", "A acquireUsage", "A releaseUsage")
        #expect(library.state.usages["text"] == nil)
        #expect(await leases(harness) == 0)
        library.send(.deleteRequested("text-fp16"))
        try await next(&events, "A delete")
        #expect(library.state.installations["text-fp16"] == ModelInstallation())
    }
}
