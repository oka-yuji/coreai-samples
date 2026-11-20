#if os(macOS)
    import Darwin
    import Foundation
    import SpeechAssets
    import Testing

    @testable import SpeechRuntime

    struct VoxCPMVerifiedCopyEvidenceTests {
        @Test("VRA-04 Verified-copy evidence accepts only the keyed copy with every selected asset and tokenizer")
        func location() throws {
            let fixture = try VoxCPMBundleFixture()
            defer { fixture.remove() }
            let raw = try VoxCPMBundleFixture.manifest(.referenceFP32)
            var contents = fixture.contents
            contents["manifest.json"] = raw
            try fixture.write(raw, to: "manifest.json")
            for name in ["main.mlirb", "main.hash", "metadata.json"] {
                let path = "artifacts/refenc-fp32.aimodel/" + name
                let data = Data(("fixture-" + name).utf8)
                contents[path] = data
                try fixture.write(data, to: path)
            }
            let files = try VoxCPMBundleInventory.files(inventory: VoxCPMBundleFixture.inventory(contents))
            let layout = try SpeechBundleLayout.voxcpm2(manifest: raw, files: files)
            let key = ValidatedSpeechBundle.verifiedKey(manifest: raw, files: files, names: .voxcpm2)
            let evidence = VoxCPMVerifiedCopyEvidence(key: key, assetPaths: layout.assets)
            func matches(_ value: ValidatedSpeechBundle) -> Bool {
                evidence.matches(value, source: fixture.source, copies: fixture.copies)
            }
            let fallback = try ValidatedSpeechBundle.open(
                fixture.source, copyingInto: fixture.copies, layout: { try .voxcpm2(manifest: $0, files: files) },
                names: .voxcpm2, inPlaceWhenCloneFails: true, clone: { _, _ in EXDEV })
            #expect(fallback.directory == fixture.source)
            #expect(!matches(fallback))
            let copy = try ValidatedSpeechBundle.open(
                fixture.source, copyingInto: fixture.copies, layout: { try .voxcpm2(manifest: $0, files: files) },
                names: .voxcpm2, inPlaceWhenCloneFails: false, clone: Self.copyDescriptor)
            #expect(matches(copy))
            #expect(!matches(.init(directory: fixture.source, assets: copy.assets, tokenizer: copy.tokenizer)))
            var swapped = copy.assets
            swapped.swapAt(0, 1)
            #expect(!matches(.init(directory: copy.directory, assets: swapped, tokenizer: copy.tokenizer)))
            var borrowed = copy.assets
            borrowed[6] = fallback.assets[6]
            #expect(!matches(.init(directory: copy.directory, assets: borrowed, tokenizer: copy.tokenizer)))
            #expect(!matches(.init(directory: copy.directory, assets: copy.assets, tokenizer: fallback.tokenizer)))
            let other = VoxCPMVerifiedCopyEvidence(key: String(repeating: "a", count: 64), assetPaths: layout.assets)
            #expect(!other.matches(copy, source: fixture.source, copies: fixture.copies))
        }

        private static func copyDescriptor(_ fd: Int32, _ target: String) -> Int32 {
            do {
                let file = FileHandle(fileDescriptor: fd, closeOnDealloc: false)
                let data = try file.readToEnd() ?? Data()
                try data.write(to: URL(fileURLWithPath: target), options: .withoutOverwriting)
                return 0
            } catch { return EIO }
        }
    }
#endif
