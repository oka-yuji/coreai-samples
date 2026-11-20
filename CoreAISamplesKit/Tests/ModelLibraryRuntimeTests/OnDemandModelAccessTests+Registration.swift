import FeatureStore
import Foundation
import Testing

@testable import ModelLibraryFeature
@testable import ModelLibraryRuntime

extension OnDemandModelAccessTests {
    @Test("Deleting and reinstalling in another window updates only previews of the affected model without a lease")
    func sharesRegistrationAcrossWindows() async throws {
        let harness = try makeHarness()
        defer { harness.close() }
        var events = harness.events.makeAsyncIterator()
        let manager = harness.scene("manager")
        harness.runner.register(manager.library)
        try await manager.download("image-fp16", &events)
        let preview = sessionless(manager, in: harness)
        let otherVariant = sessionless(manager, in: harness)
        otherVariant.send(.selectArtifact(modelID: selection.modelID, artifactID: "image-int4"))
        let untouched = otherVariant.state
        let ordinary = harness.scene("ordinary")
        harness.runner.register(ordinary.library)
        ordinary.library.send(.openDemo(residentSelection))
        let ordinarySnapshot = ordinary.library.state
        let original = try #require(harness.runner.access(for: selection, in: preview))
        let identity = original.resource.identity
        #expect(await original.validate())

        manager.library.send(.deleteRequested(identity.artifactID))
        try await next(&events, "manager delete")
        #expect(
            preview.state.selection == selection && preview.state.installations[identity.artifactID]?.phase == .missing)
        #expect(!original.isCurrent())
        #expect(await original.validate() == false)
        #expect(harness.runner.access(for: selection, in: preview) == nil)
        #expect(otherVariant.state == untouched && ordinary.library.state == ordinarySnapshot)

        try await manager.download(identity.artifactID, &events)
        let replacement = try #require(harness.runner.access(for: selection, in: preview))
        #expect(replacement.resource.registrationGeneration != original.resource.registrationGeneration)
        #expect(await replacement.validate())
        #expect(await original.validate() == false)
        #expect(otherVariant.state == untouched && ordinary.library.state == ordinarySnapshot)
        #expect(await harness.store.activeLeases(identity: identity) == 0)
    }

    @Test(
        "Ledger validation refreshes a missed deletion or replacement without repeating inspection on the revoked handle",
        arguments: [false, true])
    func validatesMissedRegistration(reinstalled: Bool) async throws {
        let harness = try makeHarness()
        defer { harness.close() }
        var events = harness.events.makeAsyncIterator()
        let manager = harness.scene("manager")
        try await manager.download("image-fp16", &events)
        let preview = sessionless(manager, in: harness)
        let original = try #require(harness.runner.access(for: selection, in: preview))
        let identity = original.resource.identity
        let isolated = Store<ModelLibraryFeature>(initialState: manager.library.state, run: harness.resources.run)

        isolated.send(.deleteRequested(identity.artifactID))
        await awaitDemoState(isolated) { $0.installations[identity.artifactID]?.phase == .missing }
        if reinstalled {
            isolated.send(.downloadRequested(identity.artifactID))
            await awaitDemoState(isolated) { $0.installations[identity.artifactID]?.phase == .ready }
        }
        let installed = try #require(isolated.state.installations[identity.artifactID])
        let registration = installed.resource?.registrationGeneration
        #expect(original.isCurrent())
        #expect(original.resource.registrationGeneration != registration)
        #expect(await original.validate() == false)
        await awaitDemoState(preview) {
            $0.installations[identity.artifactID]?.phase == installed.phase
                && $0.installations[identity.artifactID]?.resource?.registrationGeneration == registration
        }
        let restored = preview.state
        let replacement = harness.runner.access(for: selection, in: preview)
        if reinstalled {
            let replacement = try #require(replacement)
            #expect(await replacement.validate())
        } else {
            #expect(replacement == nil)
        }
        #expect(await original.validate() == false)
        #expect(preview.state == restored && restored.selection == selection)
        #expect(await harness.store.activeLeases(identity: identity) == 0)
    }

    @Test("A ledger check suspended during navigation cannot validate the old screen or reinspect the new one")
    func validateRechecksSelectionAfterWaiting() async throws {
        let harness = try makeHarness()
        defer { harness.close() }
        var events = harness.events.makeAsyncIterator()
        let manager = harness.scene("manager")
        try await manager.download("image-fp16", &events)
        let preview = sessionless(manager, in: harness)
        let access = try #require(harness.runner.access(for: selection, in: preview))
        harness.ledger.hold()
        manager.library.send(.downloadRequested("text-fp16"))
        try await next(&events, "manager download", "ledger")
        let validation = Task.immediate { await access.validate() }
        preview.send(.closeDemo)
        preview.send(.openDemo(selection))
        let replacement = preview.state
        harness.ledger.release()
        #expect(await validation.value == false)
        #expect(preview.state == replacement)
        #expect(await harness.store.activeLeases(identity: access.resource.identity) == 0)
    }
}
