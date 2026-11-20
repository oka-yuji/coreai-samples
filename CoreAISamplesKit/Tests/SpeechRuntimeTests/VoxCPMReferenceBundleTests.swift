import Foundation
import SpeechAssets
import Testing

@testable import SpeechRuntime

struct VoxCPMReferenceBundleTests {
    @Test("VRA-01 The pinned reference FP32 manifest resolves to seven ordered assets and a stateless audio encoder")
    func pinnedReference() throws {
        let url = try #require(
            Bundle.module.url(
                forResource: "voxcpm2-reference_fp32-manifest", withExtension: "json", subdirectory: "Resources"))
        let bytes = try Data(contentsOf: url)
        let manifest = try VoxCPMBundleInventory.manifest(in: bytes)
        let profile = try manifest.validate()
        #expect(profile == .referenceFP32)
        #expect(profile.assetNames == ["tslm", "ralm", "proj", "locenc", "dit", "vae", "refenc"])
        let reference = try #require(manifest.reference)
        #expect(
            reference.sampleRate == 16_000 && reference.minimumSamples == 80_000 && reference.maximumSamples == 160_000)
        let signature = try #require(manifest.assets["refenc"]?.functions["main"])
        #expect(signature.states.isEmpty)
        let layout = try SpeechBundleLayout.voxcpm2(manifest: bytes)
        #expect(layout.files.count == 25)
        #expect(layout.assets == profile.assetNames.map { "artifacts/" + $0 + "-fp32.aimodel" })
    }
}
