#if os(macOS) || os(iOS)
    import Foundation
    import SpeechAssets

    extension VoxCPMPipeline where Components == VoxCPMAssets {
        @concurrent
        static func load(bundle source: URL, copyingInto copies: URL? = nil) async throws -> (
            pipeline: Self, bundle: ValidatedSpeechBundle, statistics: VoxCPMLoadStatistics
        ) {
            let clock = ContinuousClock()
            let start = clock.now
            try Task.checkCancellation()
            var manifest = Data()
            let bundle = try ValidatedSpeechBundle.open(
                source, copyingInto: copies,
                layout: { data in
                    manifest = data
                    return try SpeechBundleLayout.voxcpm2(manifest: data)
                }, names: .voxcpm2)
            let opened = clock.now
            let contract = try VoxCPMContract(manifest: VoxCPMBundleInventory.manifest(in: manifest))
            let (assets, seconds) = try await VoxCPMAssets.load(assets: bundle.assets, contract: contract)
            let statistics = VoxCPMLoadStatistics(
                openSeconds: (opened - start) / .seconds(1), assetSeconds: seconds,
                totalSeconds: (clock.now - start) / .seconds(1))
            return (Self(components: assets, contract: contract), bundle, statistics)
        }
    }
#endif
