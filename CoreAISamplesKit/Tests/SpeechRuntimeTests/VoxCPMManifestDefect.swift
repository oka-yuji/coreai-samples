enum VoxCPMManifestDefect: CaseIterable, Sendable {
    case format, modelID, revision, configurationName, configurationPrecision, assetPrecision, sampleRate, patch
    case tokenizer, missingAsset, extraAsset, assetPath, missingFunction, extraFunction, notJSON
}
