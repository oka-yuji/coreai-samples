import Foundation
import IrodoriInputLogic

public enum IrodoriText {
    public static func normalize(_ text: String) throws -> String {
        let replacements = [
            ("\t", ""), ("[n]", ""), ("\\[n\\]", ""), ("\u{3000}", ""),
            ("\u{ff1f}", "?"), ("\u{ff01}", "!"), ("\u{2665}", "\u{2661}"), ("\u{25cf}", "\u{25cb}"),
            ("\u{25ef}", "\u{25cb}"), ("\u{3007}", "\u{25cb}"),
        ]
        var value = text
        for (old, new) in replacements { value = value.replacingOccurrences(of: old, with: new, options: .literal) }
        let removed = Set(
            ";\u{25bc}\u{2640}\u{2642}\u{300a}\u{300b}\u{226a}\u{226b}\u{e63e}\u{2460}\u{2461}\u{2462}\u{2463}\u{2464}\u{2465}"
                .unicodeScalars.map(\.value))
        let dashes: Set<UInt32> = [0x02d7, 0x2043, 0x2212, 0x23af, 0x23e4, 0x2500, 0x2501, 0x2e3a, 0x2e3b]
        let filtered = value.unicodeScalars.filter {
            !removed.contains($0.value) && !dashes.contains($0.value) && !(0x2010...0x2015).contains($0.value)
        }
        value = String(String.UnicodeScalarView(filtered))
        value = value.replacingOccurrences(of: "\u{ff5e}", with: "\u{30fc}", options: .literal).replacingOccurrences(
            of: "\u{301c}", with: "\u{30fc}", options: .literal)
        value = value.replacingOccurrences(of: "\u{2026}{3,}", with: "\u{2026}\u{2026}", options: .regularExpression)
        value = try unicode15NFKC(stripOuterBrackets(value))
        value = value.replacingOccurrences(of: "...", with: "\u{2026}", options: .literal).replacingOccurrences(
            of: "..", with: "\u{2026}", options: .literal)
        return pythonStrip(value)
    }

    private static func stripOuterBrackets(_ text: String) -> String {
        let pairs: [UInt32: UInt32] = [0x300c: 0x300d, 0x300e: 0x300f, 0xff08: 0xff09, 0x3010: 0x3011, 0x28: 0x29]
        var scalars = Array(text.unicodeScalars)
        while scalars.count >= 2 {
            guard let first = scalars.first, let last = scalars.last, pairs[first.value] == last.value else { break }
            var depth = 0
            var encloses = true
            for (index, scalar) in scalars.enumerated() {
                if scalar.value == first.value { depth += 1 } else if scalar.value == last.value { depth -= 1 }
                if depth == 0 && index < scalars.count - 1 {
                    encloses = false
                    break
                }
            }
            guard encloses && depth == 0 else { break }
            scalars.removeFirst()
            scalars.removeLast()
        }
        return String(String.UnicodeScalarView(scalars))
    }

    private static func pythonStrip(_ text: String) -> String {
        func isWhitespace(_ value: UInt32) -> Bool {
            (0x09...0x0d).contains(value) || (0x1c...0x20).contains(value)
                || (0x2000...0x200a).contains(value)
                || [0x85, 0xa0, 0x1680, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000].contains(value)
        }
        let scalars = Array(text.unicodeScalars)
        var start = 0
        var end = scalars.count
        while start < end && isWhitespace(scalars[start].value) { start += 1 }
        while end > start && isWhitespace(scalars[end - 1].value) { end -= 1 }
        return String(String.UnicodeScalarView(scalars[start..<end]))
    }
}
