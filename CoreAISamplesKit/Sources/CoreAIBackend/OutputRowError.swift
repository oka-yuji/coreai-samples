@_spi(Verification) public enum OutputRowError: Error, Equatable {
    case unreadableOutput(String)
    case notRead(String)
    case nonContiguousRows(String)
    case rowOutOfRange(row: Int, rows: Int)
}
