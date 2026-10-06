import Foundation

/// The first number in a span's text and the unit written right after it.
public struct ParsedNumber: Equatable, Sendable {
    public var value: Decimal
    /// One of MPa, kPa, degrees Celsius, the kanji for degree, mm, %, km and L, spelled as in that list.
    public var unit: String?
    /// `unit` in the form's spelling: the kanji for degree becomes degrees Celsius.
    public var normalizedUnit: String?

    public init(value: Decimal, unit: String?, normalizedUnit: String?) {
        self.value = value
        self.unit = unit
        self.normalizedUnit = normalizedUnit
    }

    /// Reads half- or full-width digits, an optional decimal part, and a unit right after them (letters in any case).
    /// Nil when the text has no digits: numbers in kana or kanji stay as written.
    public init?(parsing text: String) {
        let scalars = text.unicodeScalars.map(\.halfWidth)
        guard let start = scalars.firstIndex(where: \.isASCIIDigit) else { return nil }
        var end = scalars[start...].firstIndex { !$0.isASCIIDigit } ?? scalars.count
        if end + 1 < scalars.count, scalars[end] == ".", scalars[end + 1].isASCIIDigit {
            end = scalars[(end + 1)...].firstIndex { !$0.isASCIIDigit } ?? scalars.count
        }
        guard let value = Decimal(string: String(String.UnicodeScalarView(scalars[start..<end]))) else { return nil }
        let rest = String(String.UnicodeScalarView(scalars[end...])).lowercased().unicodeScalars
        let unit = Self.units.first { rest.starts(with: $0.lowercased().unicodeScalars) }
        self.init(value: value, unit: unit, normalizedUnit: unit.map { Self.unitSynonyms[$0] ?? $0 })
    }

    private static let units = ["MPa", "kPa", "\u{2103}", "\u{5EA6}", "mm", "%", "km", "L"]
    private static let unitSynonyms = ["\u{5EA6}": "\u{2103}"]
}

extension Unicode.Scalar {
    /// The ASCII form of a full-width ASCII variant (U+FF01...U+FF5E).
    var halfWidth: Unicode.Scalar {
        (0xFF01...0xFF5E).contains(value) ? Unicode.Scalar(value - 0xFEE0) ?? self : self
    }

    var isASCIIDigit: Bool { ("0"..."9").contains(self) }
}
