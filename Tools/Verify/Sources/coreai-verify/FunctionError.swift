enum FunctionError: Error, Equatable, CustomStringConvertible {
    case outputExists(String)
    case cannot(String, String)
    case invalidList(String)
    case noSuchFunction(String, available: [String])
    case hasStates(String, [String])
    case duplicateInput(String)
    case unknownInput(String)
    case missingInput(String)
    case unreadableType(input: String, typeName: String)
    case dtypeMismatch(input: String, asset: String, given: FunctionDtype)
    case rankMismatch(input: String, asset: Int, given: Int)
    case dimensionMismatch(input: String, axis: Int, asset: Int, given: Int)
    case notRegularFile(String)
    case fileSize(String, expected: Int, actual: Int)
    case unusableOutputName(String)

    var description: String {
        switch self {
        case .outputExists(let path):
            "The output already exists: \(path)"
        case .cannot(let action, let reason):
            "Cannot \(action): \(reason)."
        case .invalidList(let reason):
            "The inputs file is not valid: \(reason)."
        case .noSuchFunction(let name, let available):
            "The asset has no function \(name). Its functions are: \(available.joined(separator: ", "))."
        case .hasStates(let name, let states):
            "\(name) has state (\(states.joined(separator: ", "))). This command runs functions without state only."
        case .duplicateInput(let name):
            "The input \(name) is listed more than once."
        case .unknownInput(let name):
            "\(name) is not an input of the function."
        case .missingInput(let name):
            "The input \(name) of the function is not listed."
        case .unreadableType(let input, let typeName):
            "The asset describes \(input) as \"\(typeName)\", which this command cannot read."
        case .dtypeMismatch(let input, let asset, let given):
            "\(input) is \(asset) in the asset and \(given.rawValue) in the list."
        case .rankMismatch(let input, let asset, let given):
            "\(input) has \(asset) dimensions in the asset and \(given) in the list."
        case .dimensionMismatch(let input, let axis, let asset, let given):
            asset < 0
                ? "Dimension \(axis) of \(input) is dynamic and needs 1 or more, not \(given)."
                : "Dimension \(axis) of \(input) is \(asset) in the asset and \(given) in the list."
        case .notRegularFile(let label):
            "\(label) is not a regular file. Links, folders and pipes are not read."
        case .fileSize(let label, let expected, let actual):
            "\(label) has \(actual) bytes, but its shape and dtype need \(expected)."
        case .unusableOutputName(let name):
            "The output name \"\(name)\" cannot be a file name."
        }
    }
}
