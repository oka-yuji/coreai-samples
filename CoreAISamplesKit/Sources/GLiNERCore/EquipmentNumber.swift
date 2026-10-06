import Foundation

/// The words of rule N1, kept in `Resources/equipment_numbers.json` because they are Japanese: the markers of an
/// equipment name around its number (such as `No.` and `P-` before it, and the words for unit and line after it) and
/// the numeral words in kana and kanji.
struct EquipmentNumberWords: Decodable, Equatable, Sendable {
    /// Text that ends right before the numeral run of an equipment name.
    var before: [String]
    /// Text that starts right after it.
    var after: [String]
    /// The words that make a numeral run, besides half- and full-width digits.
    var numerals: [String]

    /// The file ships with the package: one that is missing or broken stops the app rather than let N1 drop nothing.
    static let bundled: EquipmentNumberWords = {
        do {
            guard
                let url = Bundle.module.url(
                    forResource: "equipment_numbers", withExtension: "json", subdirectory: "Resources")
            else { throw CocoaError(.fileNoSuchFile) }
            return try JSONDecoder().decode(EquipmentNumberWords.self, from: Data(contentsOf: url))
        } catch {
            preconditionFailure("The bundled equipment_numbers.json cannot be read: \(error)")
        }
    }()

    /// The numeral runs of a note and whether a marker touches each. A run goes on while digits and numeral words
    /// follow one another (the longest word where several start at one place); a decimal point ends it. The note and
    /// the markers are compared folded, one scalar for one, so positions do not move.
    func runs(in scalars: [Unicode.Scalar]) -> [(range: Range<Int>, marked: Bool)] {
        let words = numerals.map { Array($0.unicodeScalars) }
        var runs: [Range<Int>] = []
        var index = 0
        while index < scalars.count {
            let width =
                scalars[index].halfWidth.isASCIIDigit
                ? 1 : words.filter { scalars[index...].starts(with: $0) }.map(\.count).max() ?? 0
            if width > 0, let last = runs.last, last.upperBound == index {
                runs[runs.count - 1] = last.lowerBound..<(index + width)
            } else if width > 0 {
                runs.append(index..<(index + width))
            }
            index += max(width, 1)
        }
        let folded = scalars.map(\.folded)
        let before = self.before.map { $0.unicodeScalars.map(\.folded) }
        let after = self.after.map { $0.unicodeScalars.map(\.folded) }
        return runs.map { run in
            let marked =
                before.contains { folded[..<run.lowerBound].suffix($0.count).elementsEqual($0) }
                || after.contains { folded[run.upperBound...].starts(with: $0) }
            return (run, marked)
        }
    }
}

extension Unicode.Scalar {
    /// Rule N1's comparison form: `halfWidth` (U+FF01...U+FF5E as ASCII), then the first scalar of its lower case.
    fileprivate var folded: Unicode.Scalar {
        halfWidth.properties.lowercaseMapping.unicodeScalars.first ?? halfWidth
    }
}
