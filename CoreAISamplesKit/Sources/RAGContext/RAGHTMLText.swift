import Foundation

public enum RAGHTMLText {
    public static func plainText(fromHTML html: String) -> String {
        var text = replacing(html, from: /<!--/, to: /-->/, with: " ")
        text = replacing(text, from: /<script\b/.ignoresCase(), to: /<\/script>/.ignoresCase(), with: " ")
        text = replacing(text, from: /<style\b/.ignoresCase(), to: /<\/style>/.ignoresCase(), with: " ")
        text = replacing(text, from: /<noscript\b/.ignoresCase(), to: /<\/noscript>/.ignoresCase(), with: " ")
        text = replacing(text, from: /<(br|\/p|\/div|\/h[1-6]|\/li|\/tr|\/table)\b/.ignoresCase(), to: />/, with: "\n")
        text = replacing(text, from: /</, to: />/, with: " ")
        return condense(decodeEntities(text))
    }

    static func ranges(
        in text: String, from opening: some RegexComponent, to closing: some RegexComponent
    ) -> [Range<String.Index>] {
        var ranges: [Range<String.Index>] = []
        var start = text.startIndex
        while let open = text[start...].firstMatch(of: opening),
            let close = text[open.range.upperBound...].firstMatch(of: closing)
        {
            ranges.append(open.range.lowerBound..<close.range.upperBound)
            start = close.range.upperBound
        }
        return ranges
    }

    private static func replacing(
        _ text: String, from opening: some RegexComponent, to closing: some RegexComponent, with replacement: String
    ) -> String {
        var result = ""
        var start = text.startIndex
        for range in ranges(in: text, from: opening, to: closing) {
            result += text[start..<range.lowerBound]
            result += replacement
            start = range.upperBound
        }
        return result + text[start...]
    }

    public static func condense(_ text: String) -> String {
        text.split(separator: "\n")
            .map { $0.split(whereSeparator: \.isWhitespace).joined(separator: " ") }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    private static let namedEntities = [
        "amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "nbsp": " ", "hellip": "…",
        "mdash": "—", "ndash": "–", "rsquo": "’", "lsquo": "‘", "ldquo": "“", "rdquo": "”",
    ]

    public static func decodeEntities(_ text: String) -> String {
        guard text.contains("&") else { return text }
        return text.replacing(/&(#[xX]?[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);/) { match in
            let body = match.1
            guard body.hasPrefix("#") else { return namedEntities[String(body)] ?? String(match.0) }
            let hex = body.hasPrefix("#x") || body.hasPrefix("#X")
            let digits = body.dropFirst(hex ? 2 : 1)
            guard let code = UInt32(digits, radix: hex ? 16 : 10), let scalar = Unicode.Scalar(code) else {
                return String(match.0)
            }
            return String(Character(scalar))
        }
    }
}
