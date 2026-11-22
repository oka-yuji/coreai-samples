struct PreparedCopyAttempt: Sendable {
    let mark: PreparedCopyMark
    let use: PreparedCopyUse

    var recovering: Bool {
        mark.kind == .recovery
    }
}
