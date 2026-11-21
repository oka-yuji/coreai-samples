@_spi(Verification) public enum LogitRowError: Error, Equatable {
    case noLogits
    case nonContiguousRows
    case rowOutOfRange(row: Int, rows: Int)
}
