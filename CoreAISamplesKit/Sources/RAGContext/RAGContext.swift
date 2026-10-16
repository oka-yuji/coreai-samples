import CryptoKit
import Foundation

public struct RAGSource: Sendable, Equatable {
    public var title: String
    public var url: String
    public var date: String
    public var text: String
    public var document: String?
    public var version: String?
    public var offset: Int?

    public init(title: String, url: String, date: String = "", text: String) {
        self.title = title
        self.url = url
        self.date = date
        self.text = text
    }

    public var dateLabel: String { date.isEmpty ? "日付不明" : date }
}

public enum RAGContext {
    public static let sourceTokenBudget = 3_000
    public static let promptTokenBudget = 8_000

    public static func documentSources(
        title: String, url: String, document: String? = nil, date: String = "", text: String
    ) -> [RAGSource] {
        let version = SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
        var result: [RAGSource] = []
        var start = text.startIndex
        var offset = 0
        while start < text.endIndex {
            let end = text.index(start, offsetBy: 500, limitedBy: text.endIndex) ?? text.endIndex
            var source = RAGSource(title: title, url: url, date: date, text: String(text[start..<end]))
            source.document = document ?? url
            source.version = version
            source.offset = offset
            result.append(source)
            if end == text.endIndex { break }
            start = text.index(start, offsetBy: 400)
            offset += 400
        }
        return result
    }

    public static func instruction(today: String) -> String {
        """
        次の資料だけを根拠にして、質問に日本語で簡潔に(3文以内で)答えてください。
        - 使った資料の番号を [1] のように文中に書いてください。
        - 資料に書かれていないことは推測せず、「資料にありません」とだけ答えてください。
        - 日付は資料に書かれている日付をそのまま使ってください。
        今日は \(today) です。
        """
    }

    public static func block(_ source: RAGSource, number: Int) -> String {
        "資料[\(number)] \(source.title) / \(source.url) / \(source.dateLabel)\n\(source.text)"
    }

    public static func message(question: String, sources: [RAGSource]) -> String {
        (sources.enumerated().map { block($1, number: $0 + 1) } + ["質問: \(question)"]).joined(separator: "\n\n")
    }

    public static func prompt(question: String, sources: [RAGSource], today: String) -> String {
        instruction(today: today) + "\n\n" + message(question: question, sources: sources)
    }

    public static func citations(_ sources: [RAGSource]) -> [String] {
        sources.enumerated().map { "[\($0 + 1)] \($1.title) — \($1.url) (\($1.dateLabel))" }
    }

    public static let noSourcesReply = "Could not retrieve any sources."

    public static func pack(
        _ sources: [RAGSource], budget: Int = sourceTokenBudget, count: (String) -> Int
    ) -> [RAGSource] {
        var kept: [RAGSource] = []
        var used = 0
        for source in sources {
            let cost = count(block(source, number: 1)) + 2
            guard used + cost <= budget else { continue }
            kept.append(source)
            used += cost
        }
        return kept
    }

    public static func chunks(_ text: String, size: Int = 500, overlap: Int = 100) -> [String] {
        let units =
            text
            .split(separator: "\n", omittingEmptySubsequences: true)
            .flatMap { sentences(String($0)) }
            .flatMap { $0.count <= size ? [$0] : hardSplit($0, size: size) }
        var out: [String] = []
        var current = ""
        for unit in units {
            if current.isEmpty {
                current = unit
            } else if current.count + 1 + unit.count <= size {
                current += "\n" + unit
            } else {
                out.append(current)
                current = String(current.suffix(overlap)) + "\n" + unit
            }
        }
        if !current.isEmpty { out.append(current) }
        return out
    }

    static func hardSplit(_ text: String, size: Int) -> [String] {
        var out: [String] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            out.append(String(rest.prefix(size)))
            rest = rest.dropFirst(size)
        }
        return out
    }

    static func sentences(_ paragraph: String) -> [String] {
        var out: [String] = []
        var current = ""
        for character in paragraph {
            current.append(character)
            if "\u{3002}\u{FF0E}.!?\u{FF01}\u{FF1F}".contains(character) {
                out.append(current)
                current = ""
            }
        }
        if !current.isEmpty { out.append(current) }
        return out
    }

    private static let metaNames = [
        "article:published_time", "article:published", "og:published_time",
        "datePublished", "pubdate", "date", "dc.date", "dc.date.issued",
    ]

    public static func publishedDate(html: String) -> String? {
        for tag in html.matches(of: /<meta\b[^>]*>/.ignoresCase()) {
            let attributes = String(tag.0)
            guard
                let name = attribute("property", in: attributes) ?? attribute("name", in: attributes)
                    ?? attribute("itemprop", in: attributes),
                metaNames.contains(where: { $0.caseInsensitiveCompare(name) == .orderedSame }),
                let content = attribute("content", in: attributes),
                let date = isoDate(in: content)
            else { continue }
            return date
        }
        for tag in html.matches(of: /<time\b[^>]*>/.ignoresCase()) {
            guard let value = attribute("datetime", in: String(tag.0)),
                let date = isoDate(in: value)
            else { continue }
            return date
        }
        return nil
    }

    static func attribute(_ name: String, in tag: String) -> String? {
        guard
            let match = try? Regex("\(name)\\s*=\\s*[\"']([^\"']*)[\"']").ignoresCase()
                .firstMatch(in: tag), let range = match[1].range
        else { return nil }
        return String(tag[range])
    }

    public static func isoDate(in text: String) -> String? {
        if let match = text.firstMatch(of: /(\d{4})[-\/\u{5E74}](\d{1,2})[-\/\u{6708}](\d{1,2})/) {
            return normalize(String(match.1), String(match.2), String(match.3))
        }
        return nil
    }

    private static func normalize(_ year: String, _ month: String, _ day: String) -> String? {
        guard let m = Int(month), let d = Int(day), (1...12).contains(m), (1...31).contains(d),
            let y = Int(year), (1900...2999).contains(y)
        else { return nil }
        return String(format: "%04d-%02d-%02d", y, m, d)
    }

    public static func today(_ date: Date = Date()) -> String {
        let parts = Calendar(identifier: .gregorian).dateComponents([.year, .month, .day], from: date)
        return String(format: "%04d-%02d-%02d", parts.year ?? 0, parts.month ?? 0, parts.day ?? 0)
    }
}

extension RAGContext {
    public static func terms(_ query: String) -> [String] {
        var terms: [String] = []
        for run in query.split(whereSeparator: { !($0.isLetter || $0.isNumber) }) {
            guard run.count >= 3 else { continue }
            if run.allSatisfy(\.isASCII) {
                terms.append(String(run).lowercased())
                continue
            }
            let characters = Array(run)
            for start in 0...(characters.count - 3) {
                terms.append(String(characters[start..<(start + 3)]))
            }
        }
        var seen = Set<String>()
        return Array(terms.filter { seen.insert($0).inserted }.prefix(32))
    }

    public static func rank(_ passages: [String], query: String) -> [Int] {
        let needles = terms(query)
        guard !needles.isEmpty else { return [] }
        var scored: [(index: Int, hits: Int)] = []
        for (index, passage) in passages.enumerated() {
            let haystack = passage.lowercased()
            var hits = 0
            for needle in needles where haystack.contains(needle) { hits += 1 }
            if hits > 0 { scored.append((index, hits)) }
        }
        scored.sort { $0.hits == $1.hits ? $0.index < $1.index : $0.hits > $1.hits }
        return scored.map(\.index)
    }

    public static func measuredBlock(_ source: RAGSource) -> String { block(source, number: 1) }

    public static func select(
        _ candidates: [RAGSource], question: String, budget: Int = sourceTokenBudget,
        count: (String) -> Int,
        rank: ([String], String) -> [Int] = RAGContext.rank
    ) -> [RAGSource] {
        let ordered = rank(candidates.map(\.text), question)
        let rest = candidates.indices.filter { !ordered.contains($0) }
        return pack((ordered + rest).map { candidates[$0] }, budget: budget, count: count)
    }
}

public enum RAGDocuments {
    public static let extensions: Set<String> = ["txt", "md", "text", "markdown", "pdf"]

    public static func load(
        directory: URL, pdfText: (URL) -> String? = { _ in nil }
    ) -> [RAGSource] {
        let files =
            ((try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { extensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        return files.flatMap { url -> [RAGSource] in
            guard !Task.isCancelled else { return [] }
            let raw =
                url.pathExtension.lowercased() == "pdf"
                ? pdfText(url) : try? String(contentsOf: url, encoding: .utf8)
            guard let raw, !raw.isEmpty else { return [] }
            let name = url.lastPathComponent
            let date = RAGContext.isoDate(in: String(raw.prefix(400))) ?? ""
            return RAGContext.documentSources(
                title: name, url: name, document: url.standardizedFileURL.absoluteString, date: date, text: raw)
        }
    }
}
