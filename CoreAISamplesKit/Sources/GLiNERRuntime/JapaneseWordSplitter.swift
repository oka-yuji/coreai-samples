import GLiNERCore
import NaturalLanguage

public enum JapaneseWordSplitter {
    /// `NLTokenizer` word ranges, plus one word per non-space scalar the tokenizer skips
    /// (punctuation and symbols). Words are lower-cased; ranges count Unicode scalars.
    public static func words(in text: String) -> [WordSpan] {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.setLanguage(.japanese)
        tokenizer.string = text
        let view = text.unicodeScalars
        let scalars = Array(view)
        func scalarOffset(_ index: String.Index) -> Int { view.distance(from: view.startIndex, to: index) }
        var ranges = tokenizer.tokens(for: text.startIndex..<text.endIndex)
            .map { scalarOffset($0.lowerBound)..<scalarOffset($0.upperBound) }
            .filter { !$0.isEmpty }
        let covered = Set(ranges.joined())
        for (offset, scalar) in scalars.enumerated() where !covered.contains(offset) && !scalar.isPythonWhitespace {
            ranges.append(offset..<offset + 1)
        }
        ranges.sort { ($0.lowerBound, $0.upperBound) < ($1.lowerBound, $1.upperBound) }
        return ranges.map {
            WordSpan(text: String(String.UnicodeScalarView(scalars[$0])).lowercased(), scalarRange: $0)
        }
    }
}
