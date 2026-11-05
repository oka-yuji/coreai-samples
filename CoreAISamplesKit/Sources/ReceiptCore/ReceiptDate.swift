import Foundation

public struct ReceiptDate: Equatable, Hashable, Sendable, Codable {
    public let year: Int
    public let month: Int
    public let day: Int

    public init(year: Int, month: Int, day: Int) throws {
        let leap = year.isMultiple(of: 400) || (year.isMultiple(of: 4) && !year.isMultiple(of: 100))
        let days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        guard (1...9999).contains(year), (1...12).contains(month), (1...days[month - 1]).contains(day) else {
            throw ReceiptValidationError.invalidDate
        }
        self.year = year
        self.month = month
        self.day = day
    }

    public init(iso8601: String) throws {
        let parts = iso8601.split(separator: "-", omittingEmptySubsequences: false)
        guard parts.count == 3, parts[0].count == 4, parts[1].count == 2, parts[2].count == 2,
            parts.allSatisfy({ $0.utf8.allSatisfy { (48...57).contains($0) } }),
            let year = Int(parts[0]), let month = Int(parts[1]), let day = Int(parts[2])
        else { throw ReceiptValidationError.invalidDate }
        try self.init(year: year, month: month, day: day)
    }

    public var yearText: String { String(format: "%04d", year) }
    public var monthText: String { String(format: "%02d", month) }
    public var iso8601: String { String(format: "%04d-%02d-%02d", year, month, day) }

    public init(from decoder: any Decoder) throws {
        try self.init(iso8601: decoder.singleValueContainer().decode(String.self))
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(iso8601)
    }
}
