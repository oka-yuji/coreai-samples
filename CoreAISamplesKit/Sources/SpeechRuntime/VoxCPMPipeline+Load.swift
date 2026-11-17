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
            let opened = try open(bundle: source, copyingInto: copies)
            let openedAt = clock.now
            let (assets, seconds) = try await VoxCPMAssets.load(
                assets: opened.bundle.assets, contract: opened.contract)
            let statistics = VoxCPMLoadStatistics(
                openSeconds: (openedAt - start) / .seconds(1), assetSeconds: seconds,
                totalSeconds: (clock.now - start) / .seconds(1))
            return (Self(components: assets, contract: opened.contract), opened.bundle, statistics)
        }

        static func open(bundle source: URL, copyingInto copies: URL? = nil) throws -> VoxCPMOpenedBundle {
            var manifest = Data()
            let bundle = try ValidatedSpeechBundle.open(
                source, copyingInto: copies,
                layout: { data in
                    manifest = data
                    return try SpeechBundleLayout.voxcpm2(manifest: data)
                }, names: .voxcpm2)
            return VoxCPMOpenedBundle(
                bundle: bundle, contract: try VoxCPMContract(manifest: VoxCPMBundleInventory.manifest(in: manifest)))
        }

        @concurrent
        static func load(_ opened: VoxCPMOpenedBundle) async throws -> Self {
            let assets = try await VoxCPMAssets.load(assets: opened.bundle.assets, contract: opened.contract).assets
            return Self(components: assets, contract: opened.contract)
        }
    }
#endif
