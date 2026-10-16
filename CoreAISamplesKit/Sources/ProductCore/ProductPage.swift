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
        var document = try replace(
            #"<!--.*?-->|<(script|style)\b[^>]*>.*?</\1\s*>"#, in: html, with: " ")
        document = try removingHiddenSections(from: document)
        let title = try capture(#"<title\b[^>]*>(.*?)</title\s*>"#, in: document) ?? url.host() ?? "Product page"
        let main = try capture(#"<main\b[^>]*>(.*?)</main\s*>"#, in: document)
        let body = try capture(#"<body\b[^>]*>(.*?)</body\s*>"#, in: document)
        var text = main ?? body ?? document
        text = try replace(#"[\r\n\t]+"#, in: text, with: " ")
        text = try replace(#"<(?:h[1-6]|section)\b[^>]*>|</section\s*>"#, in: text, with: "\n\n")
        text = try replace(#"<(?:br|/p|/div|/h[1-6]|/li|/tr|/table)\b[^>]*>"#, in: text, with: "\n")
        text = try replace(#"<[^>]*>"#, in: text, with: " ")
        return try ProductSource(
            title: clean(decode(title)), text: clean(decode(text)), url: url, capturedAt: capturedAt)
    }

    private static func removingHiddenSections(from text: String) throws -> String {
        let tags = try expression(#"<(\/?)(template|nav|footer|noscript|svg)\b[^>]*>"#)
        var stack: [String] = []
        var cursor = text.startIndex
        var result = ""
        for match in tags.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            guard let range = Range(match.range, in: text), let nameRange = Range(match.range(at: 2), in: text)
            else { continue }
            let name = text[nameRange].lowercased()
            let closing = match.range(at: 1).length > 0
            if stack.isEmpty { result += text[cursor..<range.lowerBound] }
            if closing {
                if let index = stack.lastIndex(of: name) { stack.removeSubrange(index...) }
            } else {
                stack.append(name)
            }
            if stack.isEmpty { result += " " }
            cursor = range.upperBound
        }
        if stack.isEmpty { result += text[cursor...] }
        return result
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
        let text = text.replacingOccurrences(of: "\u{FEFF}", with: "").replacingOccurrences(of: "\u{200B}", with: "")
        var lines: [String] = []
        for line in text.components(separatedBy: "\n") {
            let line = line.split(whereSeparator: \.isWhitespace).joined(separator: " ")
            if !line.isEmpty || lines.last?.isEmpty == false { lines.append(line) }
        }
        return lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
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
