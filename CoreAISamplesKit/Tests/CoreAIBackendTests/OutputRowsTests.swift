import CoreAI
import Testing

@_spi(Verification) @testable import CoreAIBackend

struct OutputRowsTests {
    static let main = ["logits", "hidden_states", "tap_hidden_states"]

    private static func array(rows: Int, width: Int, rowStride: Int) -> NDArray {
        var array = NDArray(shape: [1, rows, width], scalarType: .float16, strides: [rows * rowStride, rowStride, 1])
        array.mutableView(as: Float16.self).withUnsafeMutablePointer { pointer, _, _ in
            for row in 0..<rows {
                for column in 0..<width { pointer[row * rowStride + column] = Float16(row * 100 + column) }
            }
        }
        return array
    }

    private static func values(row: Int, width: Int) -> [Float16] {
        (0..<width).map { Float16(row * 100 + $0) }
    }

    @Test(
        "A row of an output holds the width values that start at the row times the row stride, for each output of the last call"
    )
    func rowsStartAtTheRowStride() throws {
        let rows = OutputRows(
            arrays: [
                ("hidden_states", Self.array(rows: 3, width: 4, rowStride: 4)),
                ("tap_hidden_states", Self.array(rows: 3, width: 6, rowStride: 8)),
            ], isRead: true)
        #expect(
            try (0..<3).map { try rows.row($0, of: "hidden_states") } == (0..<3).map { Self.values(row: $0, width: 4) })
        #expect(
            try (0..<3).map { try rows.row($0, of: "tap_hidden_states") }
                == (0..<3).map { Self.values(row: $0, width: 6) })
    }

    @Test(
        "A row outside the rows of the last call is an error, and so is an output that the last call did not read or that has not been read yet"
    )
    func rowsOutsideTheLastCallAreErrors() throws {
        let hidden = Self.array(rows: 3, width: 4, rowStride: 4)
        let rows = OutputRows(arrays: [("hidden_states", hidden)], isRead: true)
        #expect(throws: OutputRowError.rowOutOfRange(row: 3, rows: 3)) { try rows.row(3, of: "hidden_states") }
        #expect(throws: OutputRowError.rowOutOfRange(row: -1, rows: 3)) { try rows.row(-1, of: "hidden_states") }
        #expect(throws: OutputRowError.notRead("tap_hidden_states")) { try rows.row(0, of: "tap_hidden_states") }
        #expect(throws: OutputRowError.notRead("hidden_states")) {
            try OutputRows(arrays: [("hidden_states", hidden)], isRead: false).row(0, of: "hidden_states")
        }
        #expect(throws: OutputRowError.notRead("hidden_states")) { try OutputRows().row(0, of: "hidden_states") }
    }

    @Test("Reading a row fails when the width axis of the output has a stride other than one")
    func rowsNeedAContiguousWidthAxis() {
        let strided = NDArray(shape: [1, 2, 3], scalarType: .float16, strides: [12, 6, 2])
        #expect(throws: OutputRowError.nonContiguousRows("hidden_states")) {
            try OutputRows(arrays: [("hidden_states", strided)], isRead: true).row(0, of: "hidden_states")
        }
    }

    @Test(
        "An output can be read by row only as a float16 array of one batch row with a row axis that changes with the call and a fixed width",
        arguments: [
            (.float16, [1, -1, 48], true), (.float32, [1, -1, 48], false), (.bfloat16, [1, -1, 48], false),
            (.float16, [1, -1], false), (.float16, [1, -1, 48, 1], false), (.float16, [2, -1, 48], false),
            (.float16, [-1, -1, 48], false), (.float16, [1, 4, 48], false), (.float16, [1, -1, -1], false),
            (.float16, [1, -1, 0], false),
        ] as [(NDArray.ScalarType, [Int], Bool)])
    func onlyRowShapedFloat16OutputsAreReadable(scalarType: NDArray.ScalarType, shape: [Int], readable: Bool) {
        #expect(OutputRows.readable(scalarType, shape: shape) == readable)
    }

    @Test(
        "Naming an output that the function does not have, or the logits of main, is an error, and the other outputs are accepted",
        arguments: [
            (["hidden_states", "tap_hidden_states"], main, "logits", nil),
            ([], main, "logits", nil),
            (["logits", "hidden_states"], main, "logits", "logits"),
            (["hidden", "tap_hidden_states"], main, "logits", "hidden"),
            (["tap_hidden_states"], ["tap_hidden_states"], nil, nil),
            (["hidden_states", "tap_hidden_states"], ["tap_hidden_states"], nil, "hidden_states"),
        ] as [(Set<String>, [String], String?, String?)])
    func namesMustBeOutputsOfTheFunction(
        names: Set<String>, outputNames: [String], logits: String?, unreadable: String?
    ) throws {
        if let unreadable {
            #expect(throws: OutputRowError.unreadableOutput(unreadable)) {
                try OutputRows.check(names, outputNames: outputNames, except: logits)
            }
        } else {
            try OutputRows.check(names, outputNames: outputNames, except: logits)
        }
    }
}
