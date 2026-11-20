#if os(macOS) || os(iOS)
    import CryptoKit
    import Foundation
    import SpeechAssets
    import SpeechCore

    extension SpeechBundleLayout {
        private static let voxcpm2Roots = [
            VoxCPMBundleInventory.manifestPath, VoxCPMBundleInventory.tokenizerPath, "LICENSE", "NOTICE",
        ]
        private static let voxcpm2Payload = ["main.mlirb", "main.hash", "metadata.json"]

        static func voxcpm2(manifest: Data) throws -> Self {
            let profile = try VoxCPMBundleInventory.profile(in: manifest)
            return try voxcpm2(manifest: manifest, files: VoxCPMBundleInventory.files(profile: profile))
        }

        static func voxcpm2(manifest: Data, files: [IrodoriBundleInventory.File]) throws -> Self {
            let contract = try VoxCPMBundleInventory.manifest(in: manifest)
            let profile = try contract.validate()
            let digest = SHA256.hash(data: manifest).map { String(format: "%02x", $0) }.joined()
            guard
                files.contains(where: {
                    $0.path == VoxCPMBundleInventory.manifestPath && $0.bytes == Int64(manifest.count)
                        && $0.sha256 == digest
                })
            else { throw SpeechError.invalidAsset("The VoxCPM2 manifest does not match the bundle inventory.") }
            let assets = profile.assetNames.compactMap { contract.assets[$0]?.path }
            let expected = Set(voxcpm2Roots + assets.flatMap { asset in voxcpm2Payload.map { asset + "/" + $0 } })
            guard assets.count == profile.assetNames.count, files.count == expected.count,
                Set(files.map(\.path)) == expected
            else { throw SpeechError.invalidAsset("The VoxCPM2 inventory does not list exactly the asset files.") }
            return Self(files: files, assets: assets)
        }
    }
#endif
