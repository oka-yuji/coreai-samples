import Foundation

public struct ReceiptAmount: Equatable, Hashable, Sendable, Codable {
    public let yen: Int64

    public init(text: String) throws {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
            .folding(options: .widthInsensitive, locale: Locale(identifier: "en_US_POSIX"))
        guard value.range(of: #"^(0|[1-9][0-9]*|[1-9][0-9]{0,2}(,[0-9]{3})+)$"#, options: .regularExpression) != nil,
            let yen = Int64(value.replacingOccurrences(of: ",", with: ""))
        else { throw ReceiptValidationError.invalidAmount }
        try self.init(yen: yen)
    }

    public init(yen: Int64) throws {
        guard yen >= 0 else { throw ReceiptValidationError.invalidAmount }
        self.yen = yen
    }

    public init(from decoder: any Decoder) throws {
        try self.init(text: decoder.singleValueContainer().decode(String.self))
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(String(yen))
    }
}
