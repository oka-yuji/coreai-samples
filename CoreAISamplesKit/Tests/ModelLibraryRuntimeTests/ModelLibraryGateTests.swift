import FeatureStore
import Foundation
import Testing

@testable import ModelLibraryFeature
@testable import ModelLibraryRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ModelLibraryGateTests {
    @Test("A retained gate keeps its owner alive only for one attempt and explicit abandonment still revokes it")
    func lifetime() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        var scene: LibraryScene? = harness.scene("A")
        var events = harness.events.makeAsyncIterator()
        try await scene?.download("image-fp16", &events)
        scene?.library.send(.openDemo(.imageSearch))
        let session = try #require(try await scene?.show())
        let gate = try #require(try scene?.adapter("image-search").gate(for: .imageSearch))
        var permission = gate.retainPermission()
        weak let library = scene?.library
        scene = nil
        #expect(library != nil && permission?.isPermitted() == true)
        harness.runner.abandon(owner: session.id.owner)
        #expect(permission?.isPermitted() == false && !gate.isPermitted())
        permission = nil
        #expect(library == nil && gate.retainPermission() == nil)
        await harness.runner.release(owner: session.id.owner)
        #expect(await harness.leases(session) == 0)
    }

    @Test("Releasing the last short gate retention permits owner deinitialization without a stop callback")
    func unannouncedOwnerRelease() async throws {
        let harness = try RuntimeHarness()
        defer { harness.close() }
        var scene: LibraryScene? = harness.scene("A")
        var events = harness.events.makeAsyncIterator()
        try await scene?.download("image-fp16", &events)
        scene?.library.send(.openDemo(.imageSearch))
        let session = try #require(try await scene?.show())
        let gate = try #require(try scene?.adapter("image-search").gate(for: .imageSearch))
        var permission = gate.retainPermission()
        weak let library = scene?.library
        scene = nil
        #expect(library != nil && permission?.isPermitted() == true)
        permission = nil
        #expect(library == nil && !gate.isPermitted() && gate.retainPermission() == nil)
        await harness.runner.release(owner: session.id.owner)
        #expect(await harness.leases(session) == 0)
    }
}
