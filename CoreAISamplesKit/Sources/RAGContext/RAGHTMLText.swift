import Foundation

public enum RAGHTMLText {
    public static func plainText(fromHTML html: String) -> String {
        var text = html.replacing(/<!--.*?-->/.dotMatchesNewlines(), with: " ")
        text = text.replacing(/<script\b.*?<\/script>/.dotMatchesNewlines().ignoresCase(), with: " ")
        text = text.replacing(/<style\b.*?<\/style>/.dotMatchesNewlines().ignoresCase(), with: " ")
        text = text.replacing(/<noscript\b.*?<\/noscript>/.dotMatchesNewlines().ignoresCase(), with: " ")
        text = text.replacing(/<(br|\/p|\/div|\/h[1-6]|\/li|\/tr|\/table)\b[^>]*>/.ignoresCase(), with: "\n")
        text = text.replacing(/<[^>]*>/.dotMatchesNewlines(), with: " ")
        return condense(decodeEntities(text))
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
