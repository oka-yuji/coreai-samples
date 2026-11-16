#if os(macOS) || os(iOS)
    import CryptoKit
    import Foundation
    import SpeechAssets
    import SpeechCore

    extension SpeechBundleLayout {
        static func voxcpm2(manifest: Data) throws -> Self {
            let profile = try VoxCPMBundleInventory.profile(in: manifest)
            return try voxcpm2(manifest: manifest, files: VoxCPMBundleInventory.files(profile: profile))
        }

        static func voxcpm2(manifest: Data, files: [IrodoriBundleInventory.File]) throws -> Self {
            let contract = try VoxCPMBundleInventory.manifest(in: manifest)
            let digest = SHA256.hash(data: manifest).map { String(format: "%02x", $0) }.joined()
            guard
                files.contains(where: {
                    $0.path == VoxCPMBundleInventory.manifestPath && $0.bytes == Int64(manifest.count)
                        && $0.sha256 == digest
                })
            else { throw SpeechError.invalidAsset("The VoxCPM2 manifest does not match the bundle inventory.") }
            let assets = VoxCPMBundleInventory.assetNames.compactMap { contract.assets[$0]?.path }
            guard assets.count == VoxCPMBundleInventory.assetNames.count,
                assets.allSatisfy({ asset in files.contains { $0.path.hasPrefix(asset + "/") } })
            else { throw SpeechError.invalidAsset("The VoxCPM2 inventory does not list every asset.") }
            return Self(files: files, assets: assets)
        }
    }
#endif
