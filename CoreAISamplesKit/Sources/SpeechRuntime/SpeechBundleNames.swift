#if os(macOS) || os(iOS)
    import SpeechAssets

    struct SpeechBundleNames: Equatable, Sendable {
        static let irodori = Self(
            manifest: "irodori.json", tokenizer: "tokenizer.json", verifiedFolder: "IrodoriVerified",
            model: "Irodori FP32", verifiedKeyScheme: nil)
        static let voxcpm2 = Self(
            manifest: VoxCPMBundleInventory.manifestPath, tokenizer: VoxCPMBundleInventory.tokenizerPath,
            verifiedFolder: "VoxCPM2Verified", model: "VoxCPM2", verifiedKeyScheme: "voxcpm2-verified-v1")

        let manifest: String
        let tokenizer: String
        let verifiedFolder: String
        let model: String
        let verifiedKeyScheme: String?
    }
#endif
