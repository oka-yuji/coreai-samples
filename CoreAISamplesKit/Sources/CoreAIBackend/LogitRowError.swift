@_spi(Verification) public enum LogitRowError: Error, Equatable {
    case noLogits
    case rowOutOfRange(row: Int, rows: Int)
}
