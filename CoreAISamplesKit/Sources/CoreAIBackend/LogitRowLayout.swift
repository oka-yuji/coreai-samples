struct LogitRowLayout {
    private var rows = 0
    private var rowStride = 0
    private var isContiguous = true

    mutating func update(rows: Int, strides: [Int], interleaved: Bool) {
        self.rows = rows
        rowStride = strides[1]
        isContiguous = strides.last == 1 && !interleaved
    }

    mutating func clear() {
        rows = 0
    }

    func offset(ofRow row: Int) throws(LogitRowError) -> Int {
        guard rows > 0 else { throw .noLogits }
        guard isContiguous else { throw .nonContiguousRows }
        guard (0..<rows).contains(row) else { throw .rowOutOfRange(row: row, rows: rows) }
        return row * rowStride
    }
}
