struct PreparedCopyLoad<Loaded> {
    let value: Loaded
    let recovered: Bool
    let use: PreparedCopyUse
}
