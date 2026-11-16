enum VoxCPMInventoryDefect: CaseIterable, Sendable {
    case format, count, total, digest, negativeSize, duplicatePath, absolutePath, parentPath, missingManifest, notJSON
    case combiningParentPath, nulPath, combiningName, spacePath
}
