import ImageGenerationCore
import ModelLibraryFeature

public struct ImageDeviceCalibration: Sendable {
    private let hardwareModel: String
    private let osBuild: String
    private let identity: ModelArtifactIdentity
    private let upstreamRevision: String
    private let calibratedBudget: ImageMemoryBudget

    public init(
        hardwareModel: String, osBuild: String, modelID: String, artifact: ModelArtifact, budget: ImageMemoryBudget
    ) throws {
        guard !hardwareModel.isEmpty, !osBuild.isEmpty, !modelID.isEmpty,
            artifact.isSourceValid, artifact.sourceKind == .localManifest,
            artifact.contract == "qwen-image21-mobile-v2",
            artifact.upstream?.repository == "Qwen/Qwen-Image-2.1-Turbo",
            artifact.upstream?.revision == ImageGenerationProfile.revision,
            artifact.localBundle?.manifestSHA256 == budget.manifestSHA256,
            artifact.localBundle?.profileID == budget.profileID
        else { throw ImageExecutionError.invalidResource }
        self.hardwareModel = hardwareModel
        self.osBuild = osBuild
        identity = ModelArtifactIdentity(
            modelID: modelID, artifactID: artifact.id, contract: artifact.contract, revision: budget.manifestSHA256,
            sourceKind: .localManifest)
        upstreamRevision = ImageGenerationProfile.revision
        calibratedBudget = budget
    }

    public func budget(for resource: ValidatedModelResource, hardwareModel: String, osBuild: String)
        -> ImageMemoryBudget?
    {
        guard self.hardwareModel == hardwareModel, self.osBuild == osBuild,
            resource.identity == identity, resource.upstreamRevision == upstreamRevision
        else { return nil }
        return calibratedBudget
    }
}
