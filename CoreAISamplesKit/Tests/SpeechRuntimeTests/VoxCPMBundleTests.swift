import Foundation
import SpeechCore
import Testing

@testable import SpeechAssets
@testable import SpeechRuntime

struct VoxCPMBundleTests {
    @Test("VIN-04 A VoxCPM2 bundle is checked against its inventory in a verified copy that holds only inventory files")
    func verifiedCopy() throws {
        let fixture = try VoxCPMBundleFixture()
        defer { fixture.remove() }
        for path in ["extra.txt", "artifacts/dit-fp32.aimodel/extra.bin", "artifacts/other.aimodel/main.mlirb"] {
            try fixture.write(Data("extra".utf8), to: path)
        }
        let bundle = try fixture.open()
        #expect(bundle.directory == fixture.copy)
        #expect(bundle.tokenizer == fixture.copy.appending(path: "tokenizer.json"))
        #expect(
            bundle.assets
                == VoxCPMBundleFixture.assets.map {
                    fixture.copy.appending(path: "artifacts/\($0)-fp32.aimodel", directoryHint: .isDirectory)
                })
        #expect(try fixture.copyFolders() == [VoxCPMBundleFixture.digest(fixture.manifest)])
        try #require(try fixture.copiedFiles() == Set(fixture.contents.keys))
        for (path, data) in fixture.contents {
            #expect(try Data(contentsOf: fixture.copy.appending(path: path)) == data, "\(path)")
        }
    }

    @Test(
        "VIN-04 A changed, missing or symlinked VoxCPM2 file fails the open and leaves no copy",
        arguments: VoxCPMBundleDefect.allCases)
    func rejectsDefect(_ defect: VoxCPMBundleDefect) throws {
        let fixture = try VoxCPMBundleFixture()
        defer { fixture.remove() }
        try fixture.apply(defect)
        #expect(throws: SpeechError.self) { try fixture.open() }
        #expect(try fixture.copyFolders().isEmpty)
    }

    @Test("VIN-04 The VoxCPM2 layout takes the inventory of the manifest's configuration and pins the manifest bytes")
    func productionLayout() throws {
        for profile in VoxCPMBundleProfile.allCases {
            let manifest = try VoxCPMBundleFixture.manifest(profile)
            let layout = try SpeechBundleLayout.voxcpm2(manifest: manifest)
            let suffix = profile == .fp32 ? "fp32" : "fp16"
            #expect(layout.files == (try VoxCPMBundleInventory.files(profile: profile)))
            #expect(
                layout.assets
                    == VoxCPMBundleFixture.assets.map {
                        "artifacts/\($0)-\(profile == .fp16Proj32 && $0 == "proj" ? "fp32" : suffix).aimodel"
                    })
            #expect(throws: SpeechError.self) { try SpeechBundleLayout.voxcpm2(manifest: manifest + Data(" ".utf8)) }
        }
        let root = ValidatedSpeechBundle.defaultRoot(for: .voxcpm2)
        #expect(root.lastPathComponent == "VoxCPM2Verified")
        #expect(root.deletingLastPathComponent().deletingLastPathComponent().path == URL.cachesDirectory.path)
        #expect(ValidatedSpeechBundle.defaultRoot(for: .irodori) == ValidatedSpeechBundle.defaultRoot)
    }
}
