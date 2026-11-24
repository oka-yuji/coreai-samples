import Foundation

struct FunctionTensor: Equatable {
    let name: String
    let dtype: FunctionDtype
    let shape: [Int]
    let data: Data

    static func hasRowMajorLayout(shape: [Int], strides: [Int]) -> Bool {
        guard shape.count == strides.count else { return false }
        var expected = 1
        for axis in shape.indices.reversed() {
            if shape[axis] != 1 && strides[axis] != expected { return false }
            expected *= shape[axis]
        }
        return true
    }
}
