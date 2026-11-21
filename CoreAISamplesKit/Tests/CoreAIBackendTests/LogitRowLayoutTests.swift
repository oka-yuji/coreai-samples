import Testing

@_spi(Verification) @testable import CoreAIBackend

struct LogitRowLayoutTests {
    @Test(
        "Reading a row fails before any main call and after the rows are cleared by a prefill-only read, reset, restore or unload"
    )
    func rowsAreUnavailableWithoutAMainCall() throws {
        var layout = LogitRowLayout()
        #expect(throws: LogitRowError.noLogits) { try layout.offset(ofRow: 0) }
        layout.update(rows: 2, strides: [2 * 7, 7, 1])
        #expect(try layout.offset(ofRow: 1) == 7)
        layout.clear()
        #expect(throws: LogitRowError.noLogits) { try layout.offset(ofRow: 0) }
        #expect(throws: LogitRowError.noLogits) { try layout.offset(ofRow: 1) }
    }

    @Test(
        "A row outside the last main call is an error and a row inside it starts at the row times the array's row stride"
    )
    func rowsFollowTheArrayStrides() throws {
        var layout = LogitRowLayout()
        layout.update(rows: 3, strides: [3 * 160, 160, 1])
        #expect(try [0, 1, 2].map { try layout.offset(ofRow: $0) } == [0, 160, 320])
        #expect(throws: LogitRowError.rowOutOfRange(row: 3, rows: 3)) { try layout.offset(ofRow: 3) }
        #expect(throws: LogitRowError.rowOutOfRange(row: -1, rows: 3)) { try layout.offset(ofRow: -1) }
        layout.update(rows: 1, strides: [151, 151, 1])
        #expect(try layout.offset(ofRow: 0) == 0)
        #expect(throws: LogitRowError.rowOutOfRange(row: 1, rows: 1)) { try layout.offset(ofRow: 1) }
    }
}
