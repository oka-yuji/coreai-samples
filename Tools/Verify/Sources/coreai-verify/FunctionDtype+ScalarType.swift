import CoreAI

extension FunctionDtype {
    init?(_ scalarType: NDArray.ScalarType) {
        switch scalarType {
        case .int32: self = .int32
        case .float16: self = .float16
        case .float32: self = .float32
        default: return nil
        }
    }

    var scalarType: NDArray.ScalarType {
        switch self {
        case .int32: .int32
        case .float16: .float16
        case .float32: .float32
        }
    }
}
