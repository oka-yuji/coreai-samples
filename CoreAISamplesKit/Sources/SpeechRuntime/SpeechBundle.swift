#if os(macOS) || os(iOS)
    import CryptoKit
    import Foundation
    import SpeechCore
    import SpeechAssets

    struct ValidatedSpeechBundle: Sendable {
        let directory: URL
        let assets: [URL]
        let tokenizer: URL

        static func open(_ directory: URL) throws -> Self {
            try Task.checkCancellation()
            let file = directory.appendingPathComponent("irodori.json")
            try IrodoriBundleInventory.validateManifest(Data(contentsOf: file))
            let specs = try IrodoriBundleInventory.assetSpecifications()
            let assets = try specs.map { spec in
                let asset = directory.appendingPathComponent("artifacts").appendingPathComponent(spec.stem + ".aimodel")
                for (name, expected) in spec.files {
                    let path = asset.appendingPathComponent(name)
                    let values = try path.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
                    guard values.isRegularFile == true, values.fileSize == expected.bytes,
                        try hashFile(path) == expected.sha256
                    else {
                        throw SpeechError.invalidAsset(
                            "Model files do not match the verified Irodori FP32 version: " + spec.stem)
                    }
                }
                return asset
            }
            let tokenizer = directory.appendingPathComponent("tokenizer.json")
            guard try hashFile(tokenizer) == "6a0734cf21c802169defaffe719bc2ef12bb9d0be37e54b61ed27aa89394723d" else {
                throw SpeechError.invalidAsset("The tokenizer does not match Irodori v4.1 Small MF.")
            }
            return Self(directory: directory, assets: assets, tokenizer: tokenizer)
        }
    }

    func hashFile(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while true {
            try Task.checkCancellation()
            let more = try autoreleasepool { () throws -> Bool in
                guard let bytes = try handle.read(upToCount: 4 * 1024 * 1024), !bytes.isEmpty else { return false }
                hash.update(data: bytes)
                return true
            }
            if !more { break }
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
#endif
