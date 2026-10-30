import Foundation
import ImageGenerationCore
import ImageGenerationRuntime
import ModelLibraryFeature
import Testing

struct ImageDeviceCalibrationBindingTests {
    private let artifact = QwenImage21Artifact.nativeComplex

    @Test("A calibration matches the complete artifact identity and exact hardware and OS build")
    func identity() throws {
        let budget = try syntheticBudget()
        let record = try ImageDeviceCalibration(
            hardwareModel: "iPhone18,1", osBuild: "24A446", modelID: "qwen-image21-turbo", artifact: artifact,
            budget: budget)
        #expect(record.budget(for: resource(), hardwareModel: "iPhone18,1", osBuild: "24A446") == budget)
        for field in ["model", "artifact", "contract", "revision", "upstream", "source"] {
            #expect(
                record.budget(for: resource(changing: field), hardwareModel: "iPhone18,1", osBuild: "24A446") == nil)
        }
        #expect(record.budget(for: resource(), hardwareModel: "iPhone15,4", osBuild: "24A446") == nil)
        #expect(record.budget(for: resource(), hardwareModel: "iPhone18,1", osBuild: "unknown") == nil)
        #expect(record.budget(for: resource(), hardwareModel: "", osBuild: "24A446") == nil)
    }

    @Test("A calibration cannot bind a budget for a different manifest or profile")
    func mismatchedBudget() throws {
        for changing in ["manifest", "profile"] {
            #expect(throws: ImageExecutionError.invalidResource) {
                try ImageDeviceCalibration(
                    hardwareModel: "iPhone18,1", osBuild: "24A446", modelID: "qwen-image21-turbo", artifact: artifact,
                    budget: syntheticBudget(changing: changing))
            }
        }
        #expect(throws: ImageExecutionError.invalidResource) {
            try ImageDeviceCalibration(
                hardwareModel: "", osBuild: "24A446", modelID: "qwen-image21-turbo", artifact: artifact,
                budget: syntheticBudget())
        }
    }

    @Test("Distribution and upstream metadata cannot be substituted in a local calibration")
    func sourceBoundary() throws {
        for field in ["contract", "upstream", "distribution"] {
            let changed = ModelArtifact(
                id: artifact.id, title: artifact.title,
                contract: field == "contract" ? "unknown" : artifact.contract, platforms: [.iOS],
                sizeBytes: artifact.sizeBytes,
                download: field == "distribution"
                    ? ModelDownload(repository: "example/model", revision: String(repeating: "a", count: 40)) : nil,
                upstream: field == "upstream"
                    ? ModelUpstream(repository: "example/model", revision: ImageGenerationProfile.revision)
                    : artifact.upstream, files: artifact.files,
                localBundle: field == "distribution" ? nil : artifact.localBundle)
            #expect(throws: ImageExecutionError.invalidResource) {
                try ImageDeviceCalibration(
                    hardwareModel: "iPhone18,1", osBuild: "24A446", modelID: "qwen-image21-turbo", artifact: changed,
                    budget: syntheticBudget())
            }
        }
    }

    private func resource(changing field: String? = nil) -> ValidatedModelResource {
        ValidatedModelResource(
            modelID: field == "model" ? "unknown" : "qwen-image21-turbo",
            artifactID: field == "artifact" ? "unknown" : artifact.id,
            contract: field == "contract" ? "unknown" : artifact.contract,
            revision: field == "revision" ? String(repeating: "f", count: 64) : QwenImage21Artifact.manifestSHA256,
            upstreamRevision: field == "upstream" ? nil : ImageGenerationProfile.revision,
            directory: URL(filePath: "/synthetic/model"), sizeBytes: 13_783_495_099,
            sourceKind: field == "source" ? .huggingFace : .localManifest)
    }

    private func syntheticBudget(changing field: String? = nil) throws -> ImageMemoryBudget {
        try ImageMemoryBudget(
            profileID: field == "profile" ? "unknown" : QwenImage21Artifact.profileID,
            manifestSHA256: field == "manifest"
                ? String(repeating: "f", count: 64) : QwenImage21Artifact.manifestSHA256,
            components: Dictionary(
                uniqueKeysWithValues: ImageModelComponent.allCases.map {
                    ($0, try ImageComponentMemoryBudget(loadingAdditionalBytes: 1, inferenceAdditionalBytes: 2))
                }))
    }
}
