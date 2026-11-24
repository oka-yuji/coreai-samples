import CoreAI

struct OutputRows {
    var arrays: [(name: String, array: NDArray)] = []
    var isRead = false

    static func readable(_ scalarType: NDArray.ScalarType, shape: [Int]) -> Bool {
        scalarType == .float16 && shape.count == 3 && shape[0] == 1 && shape[1] < 0 && shape[2] > 0
    }

    static func check(_ names: Set<String>, outputNames: [String], except logits: String?) throws(OutputRowError) {
        if let name = names.filter({ $0 == logits || !outputNames.contains($0) }).min() {
            throw .unreadableOutput(name)
        }
    }

    func row(_ row: Int, of name: String) throws(OutputRowError) -> [Float16] {
        guard isRead, let array = arrays.first(where: { $0.name == name })?.array else { throw .notRead(name) }
        let shape = array.shape
        let strides = array.strides
        guard shape.count == 3, strides.last == 1, array.interleaveLayout == nil else {
            throw .nonContiguousRows(name)
        }
        guard (0..<shape[1]).contains(row) else { throw .rowOutOfRange(row: row, rows: shape[1]) }
        return array.view(as: Float16.self).withUnsafePointer { pointer, _, _ in
            Array(UnsafeBufferPointer(start: pointer + row * strides[1], count: shape[2]))
        }
    }
}
