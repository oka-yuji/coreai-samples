import Foundation

struct FunctionInput: Equatable {
    let name: String
    let dtype: FunctionDtype
    let shape: [Int]
    let file: String

    var byteCount: Int {
        shape.reduce(dtype.width) { total, dimension in
            let (product, overflow) = total.multipliedReportingOverflow(by: dimension)
            return overflow ? .max : product
        }
    }

    static func read(_ data: Data) throws -> [FunctionInput] {
        guard let document = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw FunctionError.invalidList("the top level is not a JSON object")
        }
        guard Set(document.keys) == ["inputs"] else {
            throw FunctionError.invalidList("the top level must hold exactly the key inputs")
        }
        guard let entries = document["inputs"] as? [Any] else {
            throw FunctionError.invalidList("inputs is not an array")
        }
        return try entries.enumerated().map { try entry($1, at: $0) }
    }

    private static func entry(_ value: Any, at position: Int) throws -> FunctionInput {
        let place = "inputs[\(position)]"
        guard let fields = value as? [String: Any] else { throw FunctionError.invalidList("\(place) is not an object") }
        guard Set(fields.keys) == ["name", "dtype", "shape", "file"] else {
            throw FunctionError.invalidList("\(place) must hold exactly the keys name, dtype, shape and file")
        }
        guard let name = fields["name"] as? String else {
            throw FunctionError.invalidList("\(place).name is not a string")
        }
        guard let dtype = (fields["dtype"] as? String).flatMap(FunctionDtype.init(rawValue:)) else {
            throw FunctionError.invalidList("\(place).dtype must be int32, float16 or float32")
        }
        guard let items = fields["shape"] as? [Any], !items.isEmpty, items.allSatisfy({ dimension($0) != nil }) else {
            throw FunctionError.invalidList("\(place).shape must be a non-empty array of integers from 1")
        }
        guard let file = fields["file"] as? String, isInsideFolder(file) else {
            throw FunctionError.invalidList(
                "\(place).file must be a relative path inside the folder of the inputs file, without .. components")
        }
        return FunctionInput(name: name, dtype: dtype, shape: items.compactMap(dimension), file: file)
    }

    private static func dimension(_ value: Any) -> Int? {
        guard let number = value as? NSNumber, String(cString: number.objCType) == "q", number.intValue >= 1
        else { return nil }
        return number.intValue
    }

    private static func isInsideFolder(_ file: String) -> Bool {
        !file.isEmpty && !file.hasPrefix("/") && !file.contains("\u{0}")
            && !file.split(separator: "/", omittingEmptySubsequences: false).contains("..")
    }
}
