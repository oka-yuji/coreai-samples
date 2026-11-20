import ModelLibraryFeature
import ModelLibraryStorage
import SpeechAssets
import Testing

struct VoxCPMReferenceMacCatalogTests {
    @Test("VCM-01 Mac Models offers the pinned FP16 download and the distinct importable reference FP32 variant")
    func twoSources() throws {
        let catalog = try MacDemoCatalog.make()
        let model = try #require(catalog.model("voxcpm2"))
        #expect(model.defaultArtifactID == "voxcpm2-fp16")
        #expect(model.artifacts.map(\.id) == ["voxcpm2-fp16", "voxcpm2-reference_fp32"])
        let legacy = try #require(model.artifact("voxcpm2-fp16"))
        #expect(legacy == (try VoxCPMModelArtifact.makeDownload()))
        #expect(legacy.download?.revision == "afe11d61de9dd7c3609140aa040b777ef77b2555")
        #expect(legacy.download?.root == "fp16")
        #expect(ModelAcquisitionPolicy(artifact: legacy).preferredMethod == .download)
        let reference = try #require(model.artifact("voxcpm2-reference_fp32"))
        #expect(reference.contract == "voxcpm2-coreai-v2" && reference.title == "Reference voice (FP32)")
        #expect(reference.platforms == [.macOS] && reference.download == nil)
        #expect(reference.sourceKind == .localManifest && reference.isSourceValid)
        #expect(reference.localBundle?.profileID == reference.id)
        #expect(reference.sourceRevision == "5beadfe31f8b37667f3e4ef42a8451b29dca1814a4cb53a253e96d9a2fdd1b51")
        #expect(reference.files.count == 25 && reference.sizeBytes == 9_541_520_509)
        let trusted = try VoxCPMBundleInventory.files(profile: .referenceFP32)
        #expect(reference.files == trusted.map { ModelArtifactFile(path: $0.path, bytes: $0.bytes, sha256: $0.sha256) })
        #expect(ModelAcquisitionPolicy(artifact: reference).preferredMethod == .importBundle)
    }
}
