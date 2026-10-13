import Foundation

public enum ProductPage {
    public static let allowedHosts: Set<String> = [
        "apple.com", "www.apple.com", "support.apple.com",
        "microsoft.com", "www.microsoft.com", "support.microsoft.com", "learn.microsoft.com",
    ]

    public static func validatedURL(_ value: String) throws -> URL {
        guard let components = URLComponents(string: value.trimmingCharacters(in: .whitespacesAndNewlines)),
            components.scheme?.lowercased() == "https", components.user == nil, components.password == nil,
            components.port == nil || components.port == 443,
            let host = components.host?.lowercased(), allowedHosts.contains(host), let url = components.url
        else { throw ProductError.unsupportedURL }
        return url
    }

    public static func source(html: String, url: URL, capturedAt: Date) throws -> ProductSource {
        let title = try capture(#"<title\b[^>]*>(.*?)</title\s*>"#, in: html) ?? url.host() ?? "Product page"
        let main = try capture(#"<main\b[^>]*>(.*?)</main\s*>"#, in: html)
        let body = try capture(#"<body\b[^>]*>(.*?)</body\s*>"#, in: html)
        var text = main ?? body ?? html
        text = try replace(#"<!--.*?-->"#, in: text, with: " ")
        text = try replace(#"<(script|style|noscript|svg|nav|footer)\b[^>]*>.*?</\1\s*>"#, in: text, with: " ")
        text = try replace(#"<(?:br|/p|/div|/h[1-6]|/li|/tr|/table|/section)\b[^>]*>"#, in: text, with: "\n")
        text = try replace(#"<[^>]*>"#, in: text, with: " ")
        return try ProductSource(
            title: clean(decode(title)), text: clean(decode(text)), url: url, capturedAt: capturedAt)
    }

    private static func expression(_ pattern: String) throws -> NSRegularExpression {
        try NSRegularExpression(pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators])
    }

    private static func capture(_ pattern: String, in text: String) throws -> String? {
        let match = try expression(pattern).firstMatch(in: text, range: NSRange(text.startIndex..., in: text))
        guard let match, let range = Range(match.range(at: 1), in: text) else { return nil }
        return String(text[range])
    }

    private static func replace(_ pattern: String, in text: String, with value: String) throws -> String {
        try expression(pattern).stringByReplacingMatches(
            in: text, range: NSRange(text.startIndex..., in: text), withTemplate: value)
    }

    private static func clean(_ text: String) -> String {
        text.replacingOccurrences(of: "\u{FEFF}", with: "").replacingOccurrences(of: "\u{200B}", with: "")
            .split(separator: "\n").map { $0.split(whereSeparator: \.isWhitespace).joined(separator: " ") }
            .filter { !$0.isEmpty }.joined(separator: "\n")
    }

    private static func decode(_ text: String) -> String {
        let names = [
            "amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "nbsp": " ",
            "thinsp": " ", "reg": "®", "trade": "™", "copy": "©", "ndash": "–", "mdash": "—",
            "rsquo": "’", "lsquo": "‘", "rdquo": "”", "ldquo": "“", "hellip": "…",
        ]
        var result = ""
        var cursor = text.startIndex
        while cursor < text.endIndex {
            if text[cursor] == "&", let end = text[cursor...].prefix(16).firstIndex(of: ";") {
                let name = String(text[text.index(after: cursor)..<end])
                let number: UInt32? =
                    if name.lowercased().hasPrefix("#x") {
                        UInt32(name.dropFirst(2), radix: 16)
                    } else if name.hasPrefix("#") {
                        UInt32(name.dropFirst())
                    } else { nil }
                if let replacement = names[name] ?? number.flatMap(Unicode.Scalar.init).map(String.init) {
                    result += replacement
                    cursor = text.index(after: end)
                    continue
                }
            }
            result.append(text[cursor])
            cursor = text.index(after: cursor)
        }
        return result
    }
}
