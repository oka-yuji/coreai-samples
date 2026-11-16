enum VoxCPMBundleDefect: CaseIterable, Sendable {
    case changedTokenizer, changedAsset, missingFile, symlinkedFile, symlinkedFolder
}
