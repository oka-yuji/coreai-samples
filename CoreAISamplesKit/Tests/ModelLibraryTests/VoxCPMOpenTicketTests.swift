import Foundation
import Testing

@testable import ModelLibraryFeature

struct VoxCPMOpenTicketTests {
    @Test("The latest successful check and the current display grant distinguish retries at one registered location")
    func verifiesCurrentGrantAndSuccessfulInspection() throws {
        let fixture = LibraryFixture()
        var state = fixture.state(try fixture.variantCatalog(), audience: .development)
        let selected = "fp32"
        let other = "int4"
        var resource = fixture.resource(selected)
        resource.registrationGeneration = 41

        _ = ModelLibraryFeature.reduce(&state, .downloadRequested(selected))
        let validated = try #require(state.installations[selected]?.requestID)
        _ = ModelLibraryFeature.reduce(&state, .downloaded(selected, validated, resource.directory))
        _ = ModelLibraryFeature.reduce(&state, .validated(selected, validated, resource))
        try #require(state.installations[selected]?.lastSuccessfulVerification == validated)
        _ = ModelLibraryFeature.reduce(&state, .useRequested("model"))
        let firstGrant = try #require(state.usages["model"]?.pending.last)
        try #require(state.usages["model"]?.latestGrantTicket == nil)
        _ = ModelLibraryFeature.reduce(&state, .usageAcquired("model", firstGrant, resource))
        try #require(state.usages["model"]?.latestGrantTicket == firstGrant)

        _ = ModelLibraryFeature.reduce(&state, .refresh)
        let check = try #require(state.installations[selected]?.requestID)
        let unrelated = try #require(state.installations[other]?.requestID)
        try #require(state.installations[selected]?.phase == .inspecting)
        try #require(state.installations[selected]?.resource?.registrationGeneration == 41)
        _ = ModelLibraryFeature.reduce(&state, .inspected(other, unrelated, fixture.resource(other)))
        _ = ModelLibraryFeature.reduce(&state, .inspected(selected, validated, resource))
        try #require(state.installations[selected]?.lastSuccessfulVerification == validated)
        _ = ModelLibraryFeature.reduce(&state, .failed(selected, check, .modelInUse))
        try #require(state.installations[selected]?.phase == .ready)
        try #require(state.installations[selected]?.lastSuccessfulVerification == validated)

        _ = ModelLibraryFeature.reduce(&state, .refresh)
        let verified = try #require(state.installations[selected]?.requestID)
        _ = ModelLibraryFeature.reduce(&state, .inspected(selected, verified, resource))
        try #require(verified != validated && state.installations[selected]?.lastSuccessfulVerification == verified)
        try #require(state.installations[selected]?.requestID == nil)
        try #require(state.usages["model"]?.latestGrantTicket == firstGrant)
        _ = ModelLibraryFeature.reduce(&state, .useEnded("model"))
        try #require(state.usages["model"]?.latestGrantTicket == nil)
        _ = ModelLibraryFeature.reduce(&state, .useRequested("model"))
        let nextGrant = try #require(state.usages["model"]?.pending.last)
        _ = ModelLibraryFeature.reduce(&state, .usageAcquired("model", nextGrant, resource))
        try #require(firstGrant != nextGrant && state.usages["model"]?.latestGrantTicket == nextGrant)
        try #require(state.usages["model"]?.resource?.registrationGeneration == 41)
        try #require(state.installations[selected]?.lastSuccessfulVerification == verified)

        _ = ModelLibraryFeature.reduce(&state, .refresh)
        let rejected = try #require(state.installations[selected]?.requestID)
        _ = ModelLibraryFeature.reduce(&state, .inspected(selected, rejected, fixture.resource(other)))
        try #require(state.installations[selected]?.phase == .failed)
        try #require(state.installations[selected]?.lastSuccessfulVerification != rejected)
    }
}
