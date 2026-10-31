import ImageGenerationCore
import ModelLibraryFeature

extension ImageDeviceCalibration {
    public static func makeQwenImage21NativeComplex() throws -> ImageDeviceCalibration {
        let budget = try ImageMemoryBudget(
            profileID: QwenImage21Artifact.profileID,
            manifestSHA256: QwenImage21Artifact.manifestSHA256,
            components: [
                .vision: try ImageComponentMemoryBudget(
                    loadingAdditionalBytes: 3_757_047_808, inferenceAdditionalBytes: 3_435_134_976),
                .text: try ImageComponentMemoryBudget(
                    loadingAdditionalBytes: 3_731_881_984, inferenceAdditionalBytes: 3_077_570_560),
                .vaeEncoder: try ImageComponentMemoryBudget(
                    loadingAdditionalBytes: 3_537_895_424, inferenceAdditionalBytes: 3_141_533_696),
                .denoiser: try ImageComponentMemoryBudget(
                    loadingAdditionalBytes: 3_560_964_096, inferenceAdditionalBytes: 3_107_979_264),
                .vaeDecoder: try ImageComponentMemoryBudget(
                    loadingAdditionalBytes: 3_597_664_256, inferenceAdditionalBytes: 2_055_208_960),
            ])
        return try ImageDeviceCalibration(
            hardwareModel: "iPhone18,1", osBuild: "24A446", modelID: "qwen-image21-turbo",
            artifact: QwenImage21Artifact.nativeComplex, budget: budget)
    }
}
