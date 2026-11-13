import Foundation
import ImageGenerationCore
import ImageGenerationRuntime
import ModelLibraryFeature
import Testing

struct MobileImageLabCalibrationTests {
    @Test("QD-03 The public identity carries forward all five budgets without admitting the legacy identity")
    func carriedBudgets() throws {
        let calibration = try ImageDeviceCalibration.makeMobileImageLab()
        let budget = try #require(
            calibration.budget(for: resource(), hardwareModel: "iPhone18,1", osBuild: "24A446"))
        #expect(budget.manifestSHA256 == "9d61c180a8c72b67ba34e8ece6e842fec489dc98fc68b2b875fa07a1900dd3e5")
        #expect(budget.profileID == QwenImage21Artifact.profileID)
        let expected: [(ImageModelComponent, UInt64, UInt64)] = [
            (.vision, 3_757_047_808, 3_435_134_976), (.text, 3_731_881_984, 3_077_570_560),
            (.vaeEncoder, 3_537_895_424, 3_141_533_696), (.denoiser, 3_560_964_096, 3_107_979_264),
            (.vaeDecoder, 3_597_664_256, 2_055_208_960),
        ]
        #expect(budget.components.count == expected.count)
        for (component, loading, inference) in expected {
            #expect(budget.components[component]?.loadingAdditionalBytes == loading)
            #expect(budget.components[component]?.inferenceAdditionalBytes == inference)
        }
        let legacy = try ImageDeviceCalibration.makeQwenImage21NativeComplex()
        #expect(legacy.budget(for: resource(), hardwareModel: "iPhone18,1", osBuild: "24A446") == nil)
        #expect(
            calibration.budget(for: resource(legacy: true), hardwareModel: "iPhone18,1", osBuild: "24A446") == nil)
        for field in ["model", "artifact", "contract", "manifest", "source", "upstream"] {
            #expect(
                calibration.budget(for: resource(changing: field), hardwareModel: "iPhone18,1", osBuild: "24A446")
                    == nil)
        }
        for (hardware, build) in [("unknown", "24A446"), ("iPhone18,1", "unknown"), ("iPhone18,1", "27.0.1")] {
            #expect(calibration.budget(for: resource(), hardwareModel: hardware, osBuild: build) == nil)
        }
    }

    @Test("QD-03 A carried budget still rejects an unknown profile or manifest and a missing component")
    func invalidBudget() throws {
        let calibration = try ImageDeviceCalibration.makeMobileImageLab()
        let budget = try #require(
            calibration.budget(for: resource(), hardwareModel: "iPhone18,1", osBuild: "24A446"))
        for field in ["profile", "manifest"] {
            let changed = try ImageMemoryBudget(
                profileID: field == "profile" ? "unknown" : budget.profileID,
                manifestSHA256: field == "manifest" ? String(repeating: "f", count: 64) : budget.manifestSHA256,
                components: budget.components)
            #expect(throws: ImageExecutionError.invalidResource) {
                try ImageDeviceCalibration(
                    hardwareModel: "iPhone18,1", osBuild: "24A446", modelID: "qwen-image21-turbo",
                    artifact: MobileImageLabArtifact.published, budget: changed)
            }
        }
        var incomplete = budget.components
        incomplete[.vision] = nil
        #expect(throws: ImageExecutionError.budgetNotConfigured) {
            try ImageMemoryBudget(
                profileID: budget.profileID, manifestSHA256: budget.manifestSHA256, components: incomplete)
        }
    }

    private func resource(legacy: Bool = false, changing field: String? = nil) -> ValidatedModelResource {
        let artifact = legacy ? QwenImage21Artifact.nativeComplex : MobileImageLabArtifact.published
        return ValidatedModelResource(
            modelID: field == "model" ? "unknown" : "qwen-image21-turbo",
            artifactID: field == "artifact" ? "unknown" : artifact.id,
            contract: field == "contract" ? "unknown" : artifact.contract,
            revision: field == "manifest" ? String(repeating: "f", count: 64) : artifact.sourceRevision ?? "",
            upstreamRevision: field == "upstream" ? nil : artifact.upstream?.revision,
            directory: URL(filePath: "/synthetic/image"), sizeBytes: artifact.sizeBytes ?? 0,
            sourceKind: field == "source" ? .huggingFace : .localManifest)
    }
}
