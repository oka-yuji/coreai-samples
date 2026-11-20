import Foundation
import SpeechCore
import Testing

@testable import SpeechAssets
@testable import SpeechRuntime

struct VoxCPMBundleTests {
    static let referenceOrder = ["tslm", "ralm", "proj", "locenc", "dit", "vae", "refenc"]

    static func referenceContents() throws -> [String: Data] {
        var contents = [
            "manifest.json": try VoxCPMBundleFixture.manifest(.referenceFP32), "tokenizer.json": Data("tokenizer".utf8),
            "LICENSE": Data("license".utf8), "NOTICE": Data("notice".utf8),
        ]
        for asset in referenceOrder {
            for file in ["main.mlirb", "main.hash", "metadata.json"] {
                contents["artifacts/\(asset)-fp32.aimodel/\(file)"] = Data("\(asset)/\(file)".utf8)
            }
        }
        return contents
    }

    static func write(_ contents: [String: Data], into folder: URL) throws {
        for (path, data) in contents {
            let url = folder.appending(path: path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url)
        }
    }

    static func regularFiles(in folder: URL) throws -> Set<String> {
        var result: Set<String> = []
        for path in try FileManager.default.subpathsOfDirectory(atPath: folder.path) {
            var status = stat()
            if lstat(folder.appending(path: path).path, &status) == 0, status.st_mode & S_IFMT == S_IFREG {
                result.insert(path)
            }
        }
        return result
    }

    static func openReference(_ source: URL, into copies: URL, files: [IrodoriBundleInventory.File]) throws
        -> ValidatedSpeechBundle
    {
        try ValidatedSpeechBundle.open(
            source, copyingInto: copies, layout: { try SpeechBundleLayout.voxcpm2(manifest: $0, files: files) },
            names: .voxcpm2, inPlaceWhenCloneFails: false)
    }

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
        #expect(try fixture.copyFolders() == [fixture.key])
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
            let reference = profile == .referenceFP32
            let suffix = profile == .fp32 || reference ? "fp32" : "fp16"
            let assets = reference ? Self.referenceOrder : VoxCPMBundleFixture.assets
            #expect(layout.files == (try VoxCPMBundleInventory.files(profile: profile)))
            #expect(layout.files.count == (reference ? 25 : 22), "\(profile)")
            #expect(
                layout.assets
                    == assets.map {
                        "artifacts/\($0)-\(profile == .fp16Proj32 && $0 == "proj" ? "fp32" : suffix).aimodel"
                    })
            #expect(throws: SpeechError.self) { try SpeechBundleLayout.voxcpm2(manifest: manifest + Data(" ".utf8)) }
        }
        let root = ValidatedSpeechBundle.defaultRoot(for: .voxcpm2)
        #expect(root.lastPathComponent == "VoxCPM2Verified")
        #expect(root.deletingLastPathComponent().deletingLastPathComponent().path == URL.cachesDirectory.path)
        #expect(ValidatedSpeechBundle.defaultRoot(for: .irodori) == ValidatedSpeechBundle.defaultRoot)
    }

    @Test("VRA-03 A reference FP32 bundle opens as a verified copy of its 25 listed files with seven ordered URLs")
    func referenceCopy() throws {
        let fixture = try VoxCPMBundleFixture()
        defer { fixture.remove() }
        let source = fixture.base.appending(path: "reference", directoryHint: .isDirectory)
        let contents = try Self.referenceContents()
        try Self.write(contents, into: source)
        let extra = Data("extra".utf8)
        try Self.write(["unlisted.txt": extra, "artifacts/refenc-fp32.aimodel/extra.bin": extra], into: source)
        let files = try VoxCPMBundleInventory.files(inventory: VoxCPMBundleFixture.inventory(contents))
        let manifest = try VoxCPMBundleFixture.manifest(.referenceFP32)
        let key = ValidatedSpeechBundle.verifiedKey(manifest: manifest, files: files, names: .voxcpm2)
        let copy = fixture.copies.appending(path: key, directoryHint: .isDirectory)
        let bundle = try Self.openReference(source, into: fixture.copies, files: files)
        #expect(files.count == 25 && bundle.directory == copy)
        #expect(
            bundle.assets
                == Self.referenceOrder.map {
                    copy.appending(path: "artifacts/\($0)-fp32.aimodel", directoryHint: .isDirectory)
                })
        #expect(bundle.tokenizer == copy.appending(path: "tokenizer.json"))
        #expect(try Self.regularFiles(in: copy) == Set(contents.keys))
        #expect(try Self.openReference(source, into: fixture.copies, files: files).directory == copy)
        #expect(try fixture.copyFolders() == [key])
    }

    @Test(
        "VRA-01 A reference inventory that drops, renames, adds or repeats a file or mixes versions is rejected",
        arguments: [
            "missingHash", "missingNotice", "renamedHash", "extraFile", "duplicate", "legacyInventory",
            "legacyManifest",
        ])
    func rejectsReferenceInventory(_ defect: String) throws {
        let files = try VoxCPMBundleInventory.files(inventory: VoxCPMBundleFixture.inventory(Self.referenceContents()))
        let manifest = try VoxCPMBundleFixture.manifest(.referenceFP32)
        #expect(try SpeechBundleLayout.voxcpm2(manifest: manifest, files: files).files == files)
        let hash = "artifacts/refenc-fp32.aimodel/main.hash"
        var listed = files
        var selected = manifest
        switch defect {
        case "missingHash": listed.removeAll { $0.path == hash }
        case "missingNotice": listed.removeAll { $0.path == "NOTICE" }
        case "renamedHash":
            listed = files.map {
                $0.path == hash
                    ? IrodoriBundleInventory.File(
                        path: "artifacts/refenc-fp32.aimodel/other.hash", bytes: $0.bytes, sha256: $0.sha256) : $0
            }
        case "extraFile":
            let extra = Data("extra".utf8)
            listed.append(
                IrodoriBundleInventory.File(
                    path: "artifacts/refenc-fp32.aimodel/extra.bin", bytes: Int64(extra.count),
                    sha256: VoxCPMBundleFixture.digest(extra)))
        case "duplicate": listed.append(files[0])
        case "legacyInventory": listed = try VoxCPMBundleInventory.files(profile: .fp32)
        default: selected = try VoxCPMBundleFixture.manifest(.fp32)
        }
        #expect(throws: SpeechError.self) { try SpeechBundleLayout.voxcpm2(manifest: selected, files: listed) }
    }

    @Test("VRA-01 A legacy inventory must still list exactly its 22 files")
    func legacyInventoryFiles() throws {
        let fixture = try VoxCPMBundleFixture()
        defer { fixture.remove() }
        let files = try VoxCPMBundleInventory.files(inventory: fixture.inventory())
        #expect(try SpeechBundleLayout.voxcpm2(manifest: fixture.manifest, files: files).files.count == 22)
        let encoder = try VoxCPMBundleInventory.files(profile: .referenceFP32).filter {
            $0.path.hasPrefix("artifacts/refenc-fp32.aimodel/")
        }
        for listed in [files + encoder, files.filter { $0.path != "NOTICE" }] {
            #expect(throws: SpeechError.self, "\(listed.count)") {
                try SpeechBundleLayout.voxcpm2(manifest: fixture.manifest, files: listed)
            }
        }
    }

    @Test(
        "VRA-03 A changed, missing or linked reference encoder file fails the open and leaves no copy",
        arguments: ["changed", "missing", "linkedFile", "linkedFolder"])
    func rejectsReferenceFile(_ defect: String) throws {
        let fixture = try VoxCPMBundleFixture()
        defer { fixture.remove() }
        let source = fixture.base.appending(path: "reference", directoryHint: .isDirectory)
        let contents = try Self.referenceContents()
        try Self.write(contents, into: source)
        let files = try VoxCPMBundleInventory.files(inventory: VoxCPMBundleFixture.inventory(contents))
        let package = source.appending(path: "artifacts/refenc-fp32.aimodel", directoryHint: .isDirectory)
        let model = package.appending(path: "main.mlirb")
        let manager = FileManager.default
        switch defect {
        case "changed":
            var data = try Data(contentsOf: model)
            data[0] ^= 1
            try data.write(to: model)
        case "missing": try manager.removeItem(at: model)
        case "linkedFile":
            let outside = fixture.base.appending(path: "outside.mlirb")
            try manager.moveItem(at: model, to: outside)
            try manager.createSymbolicLink(at: model, withDestinationURL: outside)
        default:
            let outside = fixture.base.appending(path: "outside.aimodel", directoryHint: .isDirectory)
            try manager.moveItem(at: package, to: outside)
            try manager.createSymbolicLink(at: package, withDestinationURL: outside)
        }
        #expect(throws: SpeechError.self) { try Self.openReference(source, into: fixture.copies, files: files) }
        #expect(try fixture.copyFolders().isEmpty)
    }

    @Test("VRA-03 Opening bundles names each profile and enables reference audio only for reference FP32")
    func referenceOpen() async throws {
        let directory = URL(filePath: "/unused-model")
        var names: [VoxCPMBundleProfile: String] = [:]
        for profile in VoxCPMBundleProfile.allCases {
            let runtime = VoxCPMRuntime(dependencies: FakeVoxCPMDriver().dependencies(profile: profile))
            let model = try await runtime.open(directory)
            #expect(
                model.supportsReferenceAudio == (profile == .referenceFP32) && model.directory == directory,
                "\(profile)")
            names[profile] = model.name
            await runtime.release()
        }
        #expect(names[.referenceFP32] == "VoxCPM2 (FP32, reference audio)")
        #expect(Set(names.values).count == VoxCPMBundleProfile.allCases.count)
    }
}
