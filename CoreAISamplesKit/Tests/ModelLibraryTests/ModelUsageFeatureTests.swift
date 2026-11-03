import FeatureStore
import Foundation
import Testing

@testable import ModelLibraryFeature

struct ModelUsageFeatureTests {
    typealias Operation = ModelLibraryFeature.Operation

    private let fixture = LibraryFixture()

    private func reduce(_ state: inout ModelLibraryState, _ action: ModelLibraryAction) -> [Operation] {
        ModelLibraryFeature.reduce(&state, action)
    }

    private func installed() throws -> ModelLibraryState {
        var state = fixture.state(try fixture.variantCatalog())
        try fixture.install(&state, "fp32")
        return state
    }

    private func acquisition(_ operations: [Operation]) -> LibraryTicket? {
        guard operations.count == 1, case .run(.acquireUsage(let identity, let ticket), .none) = operations[0],
            identity == fixture.resource("fp32").identity
        else { return nil }
        return ticket
    }

    @Test("USE-01 Each use of a ready artifact asks for its own lease and each ended use releases one lease")
    func pairsUsesWithLeases() throws {
        var state = try installed()
        let resource = fixture.resource("fp32")
        let first = try #require(acquisition(reduce(&state, .useRequested("model"))))
        #expect(state.usages["model"]?.isActive == true && state.usages["model"]?.resource == nil)
        #expect(reduce(&state, .usageAcquired("model", first, resource)).isEmpty)
        #expect(state.usages["model"]?.resource == resource)
        let second = try #require(acquisition(reduce(&state, .useRequested("model"))))
        #expect(second != first)
        #expect(reduce(&state, .usageAcquired("model", second, resource)).isEmpty)

        #expect(reduce(&state, .useEnded("model")) == [.run(.releaseUsage(resource.identity, second), id: nil)])
        #expect(state.usages["model"]?.isActive == true && state.usages["model"]?.resource == resource)
        #expect(reduce(&state, .useEnded("model")) == [.run(.releaseUsage(resource.identity, first), id: nil)])
        #expect(state.usages["model"] == nil)
        #expect(reduce(&state, .useEnded("model")).isEmpty)
        #expect(state.installations["fp32"]?.phase == .ready)
    }

    @Test("USE-01 A use requested before the artifact is ready waits and asks for the lease once it becomes ready")
    func servesWhenReady() throws {
        var state = fixture.state(try fixture.variantCatalog())
        #expect(reduce(&state, .useRequested("model")).isEmpty)
        #expect(state.usages["model"]?.isActive == false)
        #expect(reduce(&state, .downloadRequested("fp32")).count == 1)
        let request = try #require(state.installations["fp32"]?.requestID)
        _ = reduce(&state, .downloaded("fp32", request, URL(filePath: "/fixture/fp32")))
        #expect(state.usages["model"]?.isActive == false)
        let ticket = try #require(acquisition(reduce(&state, .validated("fp32", request, fixture.resource("fp32")))))
        #expect(state.usages["model"]?.isActive == true)
        #expect(reduce(&state, .usageAcquired("model", ticket, fixture.resource("fp32"))).isEmpty)
        #expect(state.usages["model"]?.resource == fixture.resource("fp32"))
    }

    @Test(
        "USE-01 A lease that arrives after its use ended or for another location is returned, and an unknown one is ignored"
    )
    func releasesLateLeases() throws {
        var state = try installed()
        let resource = fixture.resource("fp32")
        let ticket = try #require(acquisition(reduce(&state, .useRequested("model"))))
        #expect(reduce(&state, .useEnded("model")).isEmpty)
        #expect(state.usages["model"]?.isActive == true)
        #expect(
            reduce(&state, .usageAcquired("model", ticket, resource))
                == [.run(.releaseUsage(resource.identity, ticket), id: nil)])
        #expect(state.usages["model"] == nil)
        #expect(reduce(&state, .usageAcquired("model", ticket, resource)).isEmpty)

        let other = try #require(acquisition(reduce(&state, .useRequested("model"))))
        let moved = ValidatedModelResource(
            modelID: "model", artifactID: "fp32", contract: "fixture-fp32",
            revision: LibraryFixture.distributionRevision, upstreamRevision: LibraryFixture.upstreamRevision,
            directory: URL(filePath: "/elsewhere"), sizeBytes: 4)
        let returned = reduce(&state, .usageAcquired("model", other, moved))
        #expect(returned.first == .run(.releaseUsage(moved.identity, other), id: nil))
        let retry = try #require(acquisition(Array(returned.dropFirst())))
        #expect(retry != other && retry != ticket)
        #expect(state.usages["model"]?.resource == nil && state.usages["model"]?.isActive == true)
    }

    @Test("USE-01 A refused lease for an artifact that still looks ready is not retried until it is checked again")
    func refusalWaitsForInspection() throws {
        var state = try installed()
        let ticket = try #require(acquisition(reduce(&state, .useRequested("model"))))
        #expect(reduce(&state, .usageAcquired("model", ticket, nil)).isEmpty)
        #expect(state.usages["model"]?.isActive == false && state.usages["model"]?.resource == nil)
        #expect(reduce(&state, .dismissFailure).isEmpty)

        let inspections = reduce(&state, .refresh)
        guard inspections.count == 1, case .run(.inspect(_, _, let request), _) = inspections[0] else {
            Issue.record("Expected one inspection, got \(inspections)")
            return
        }
        let retry = try #require(
            acquisition(reduce(&state, .inspected("fp32", request, fixture.resource("fp32")))))
        #expect(retry != ticket)
        #expect(reduce(&state, .usageAcquired("model", retry, fixture.resource("fp32"))).isEmpty)
        #expect(state.usages["model"]?.resource == fixture.resource("fp32"))
    }

    @Test(
        "USE-02 A deletion refused because the model is in use keeps the ready state and its location, while other failures still drop it"
    )
    func refusedDeletionKeepsResource() throws {
        var state = try installed()
        let resource = fixture.resource("fp32")
        let deletion = reduce(&state, .deleteRequested("fp32"))
        let request = try #require(state.installations["fp32"]?.requestID)
        #expect(deletion == [.run(.delete(resource, request), id: .resource("fp32"))])
        #expect(reduce(&state, .failed("fp32", request, .modelInUse)).isEmpty)
        #expect(state.installations["fp32"]?.phase == .ready && state.installations["fp32"]?.resource == resource)
        #expect(state.installations["fp32"]?.failure == .modelInUse && state.installations["fp32"]?.requestID == nil)

        _ = reduce(&state, .deleteRequested("fp32"))
        let next = try #require(state.installations["fp32"]?.requestID)
        _ = reduce(&state, .failed("fp32", next, .operation("Disk error")))
        #expect(state.installations["fp32"]?.phase == .failed && state.installations["fp32"]?.resource == nil)
        #expect(state.installations["fp32"]?.failure == .operation("Disk error"))
    }
}
