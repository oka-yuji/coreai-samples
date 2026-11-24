import Foundation

struct FunctionTensorType: Equatable {
    let dtype: String
    let dimensions: [Int]

    init?(typeName: String) {
        let prefix = "NDArray ("
        guard typeName.hasPrefix(prefix), typeName.hasSuffix(")") else { return nil }
        let body = typeName.dropFirst(prefix.count).dropLast()
        guard let comma = body.firstIndex(of: ",") else { return nil }
        var dimensions: [Int] = []
        for part in body[body.index(after: comma)...].split(separator: "\u{D7}", omittingEmptySubsequences: false) {
            let text = part.trimmingCharacters(in: .whitespaces)
            if text == "?" {
                dimensions.append(-1)
            } else if let value = Int(text), text.allSatisfy({ $0.isASCII && $0.isNumber }) {
                dimensions.append(value)
            } else {
                return nil
            }
        }
        dtype = body[..<comma].trimmingCharacters(in: .whitespaces)
        self.dimensions = dimensions
    }
}
