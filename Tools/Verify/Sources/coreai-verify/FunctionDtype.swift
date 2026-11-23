enum FunctionDtype: String, Codable, CaseIterable {
    case int32
    case float16
    case float32

    var width: Int {
        switch self {
        case .int32, .float32: 4
        case .float16: 2
        }
    }

    init?(assetName: String) {
        switch assetName {
        case "Int32": self = .int32
        case "Float16": self = .float16
        case "Float32": self = .float32
        default: return nil
        }
    }
}
