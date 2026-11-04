import CoreAI

enum TypedDecisionOutput: String, Sendable {
    case allTokens = "hidden_states"
    case lastToken = "last_token_hidden_state"

    func shape(tokens: Int, width: Int) -> [Int] {
        [1, self == .allTokens ? tokens : 1, width]
    }

    func accepts(shape: [Int], type: NDArray.ScalarType, interleaved: Bool, width: Int) -> Bool {
        guard width > 0, type == .float16, !interleaved, shape.count == 3,
            shape[0] == 1, shape[2] == width
        else { return false }
        return self == .allTokens ? shape[1] < 0 : shape[1] == 1
    }

    func values(from array: NDArray, width: Int) throws -> [Float] {
        let actual = array.shape
        guard width > 0, array.scalarType == .float16, array.interleaveLayout == nil,
            actual.count == 3, actual[1] > 0, actual == shape(tokens: actual[1], width: width)
        else { throw TypedDecisionError.invalidBundle("The hidden-state output has an unsupported shape or type.") }
        var values = [Float](repeating: 0, count: width)
        array.view(as: Float16.self).withUnsafePointer { pointer, dimensions, strides in
            let row = (dimensions[1] - 1) * strides[1]
            for column in 0..<width {
                values[column] = Float(pointer[row + column * strides[2]])
            }
        }
        return values
    }
}
