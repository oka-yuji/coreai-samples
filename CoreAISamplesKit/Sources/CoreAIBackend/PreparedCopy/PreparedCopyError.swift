enum PreparedCopyError: Error, Equatable {
    case preparing, recoveryFailed, reloadRequired, inUse
}
