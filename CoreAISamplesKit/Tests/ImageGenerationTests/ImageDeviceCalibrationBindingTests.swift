import Foundation
import ImageGenerationCore
import ImageGenerationRuntime
import ModelLibraryFeature
import Testing

struct ImageDeviceCalibrationBindingTests {
    private let artifact = QwenImage21Artifact.nativeComplex

    @Test("The registered Qwen calibration matches measured memory bounds and the ordinary catalog")
    func productionCalibration() throws {
        let catalog = try IOSDemoCatalog.make()
        let model = try #require(catalog.model("qwen-image21-turbo"))
        #expect(model.artifacts == [artifact])
        #expect(artifact.download == nil && artifact.sourceKind == .localManifest)
        let calibration = try ImageDeviceCalibration.makeQwenImage21NativeComplex()
        let budget = try #require(
            calibration.budget(for: resource(), hardwareModel: "iPhone18,1", osBuild: "24A446"))
        #expect(budget.profileID == QwenImage21Artifact.profileID)
        #expect(budget.manifestSHA256 == QwenImage21Artifact.manifestSHA256)
        let witnesses: [(ImageModelComponent, ImageMemoryPhase, UInt64, UInt64)] = [
            (.denoiser, .inference, 842_718_784, 3_681_815_832),
            (.denoiser, .loading, 385_944_112, 3_677_752_600),
            (.text, .inference, 872_913_392, 3_681_815_832),
            (.text, .loading, 219_089_480, 3_681_815_832),
            (.vaeDecoder, .inference, 1_895_304_448, 3_681_815_832),
            (.vaeDecoder, .loading, 353_634_840, 3_681_815_832),
            (.vaeEncoder, .inference, 804_686_696, 3_677_752_600),
            (.vaeEncoder, .loading, 408_963_656, 3_677_752_600),
            (.vision, .inference, 515_313_040, 3_681_815_832),
            (.vision, .loading, 193_628_720, 3_681_815_832),
        ]
        let mebibyte: UInt64 = 1_048_576
        for (component, phase, before, terminalPeak) in witnesses {
            try #require(terminalPeak >= before)
            let expected = ((terminalPeak - before + 256 * mebibyte + mebibyte - 1) / mebibyte) * mebibyte
            let value = try #require(budget.components[component])
            let actual = phase == .loading ? value.loadingAdditionalBytes : value.inferenceAdditionalBytes
            #expect(actual == expected)
        }
        #expect(calibration.budget(for: resource(), hardwareModel: "iPhone15,4", osBuild: "24A446") == nil)
        #expect(calibration.budget(for: resource(), hardwareModel: "iPhone18,1", osBuild: "unknown") == nil)
    }

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
