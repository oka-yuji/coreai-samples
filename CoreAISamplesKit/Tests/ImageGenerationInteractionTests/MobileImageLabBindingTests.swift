import Foundation
import ImageGenerationFeature
import ImageGenerationRuntime
import ModelLibraryFeature
import Testing

@MainActor
struct MobileImageLabBindingTests {
    @Test("QD-06 Public service construction binds scoped permission without generating an image")
    func publicBinding() async throws {
        let artifact = MobileImageLabArtifact.published
        let resource = ValidatedModelResource(
            modelID: "qwen-image21-turbo", artifactID: artifact.id, contract: artifact.contract,
            revision: MobileImageLabArtifact.manifestSHA256, upstreamRevision: artifact.upstream?.revision,
            directory: URL(filePath: "/synthetic/public-image"), sizeBytes: 13_783_495_057, sourceKind: .localManifest)
        let calibration = try ImageDeviceCalibration.makeMobileImageLab()
        let service = ImageInteractionService()
        var calls = 0
        var constructionPermission: Bool?
        var permission: ManagedImageGeneration.Permission?
        let managed = ManagedImageGeneration { session, allowed in
            guard session.resource == resource,
                calibration.budget(for: session.resource, hardwareModel: "iPhone18,1", osBuild: "24A446") != nil
            else { return nil }
            calls += 1
            constructionPermission = allowed()
            permission = allowed
            return service
        }
        let selection = DemoSelection(modelID: resource.modelID, demoID: "image-generation")
        let session = ModelDemoSession(
            id: LibraryTicket(owner: UUID(), number: 1), selection: selection, resource: resource)
        #expect(managed.store(for: session, isPermitted: { false }) == nil && calls == 0)
        let permitted: ManagedImageGeneration.Permission = { true }
        let store = try #require(managed.store(for: session, isPermitted: permitted))
        #expect(calls == 1 && constructionPermission == false && permission?() == true)
        #expect(managed.store(for: session, isPermitted: { true }) === store && calls == 1)
        #expect(store.state.modelName == "Mobile Image Lab" && !store.state.presented)
        #expect(service.requests.isEmpty)
        let other = ModelDemoSession(
            id: LibraryTicket(owner: UUID(), number: 1), selection: selection, resource: resource)
        try await managed.release(other)
        #expect(service.releases == 0 && permission?() == true)
        try await managed.release(session)
        #expect(service.releases == 1 && permission?() == false && service.requests.isEmpty)
    }

    @Test("QD-01 Preview unsupported and ready states all use the neutral primary name")
    func displayName() {
        for availability: ImageGenerationState.Availability in [.needsModel, .unsupported, .ready] {
            let state = ImageGenerationState(owner: UUID(), availability: availability)
            #expect(state.modelName == "Mobile Image Lab" && !state.canGenerate)
        }
    }
}
