import Foundation
import RAGContext
import Testing

private func fakeCount(_ text: String) -> Int { max(1, text.count / 2) }

@Suite("RAG context")
struct RAGContextSuite {
    @Test("the splitter keeps pieces under the size and overlaps them")
    func chunkBoundaries() {
        let text = (1...40).map { "\u{3053}\u{308C}\u{306F}\($0)\u{756A}\u{76EE}\u{306E}\u{6587}\u{3067}\u{3059}\u{3002}" }.joined()
        let pieces = RAGContext.chunks(text, size: 120, overlap: 30)
        #expect(pieces.count > 1)
        for piece in pieces { #expect(piece.count <= 120 + 30) }
        for (previous, next) in zip(pieces, pieces.dropFirst()) {
            let head = next.split(separator: "\n").first.map(String.init) ?? next
            #expect(previous.hasSuffix(head), "\(head.debugDescription) is not the tail of the previous piece")
        }
        for number in [1, 17, 40] {
            #expect(pieces.contains { $0.contains("\u{3053}\u{308C}\u{306F}\(number)\u{756A}\u{76EE}\u{306E}\u{6587}\u{3067}\u{3059}\u{3002}") })
        }
    }

    @Test("a paragraph with no sentence breaks is cut at the size")
    func hardSplit() {
        let pieces = RAGContext.chunks(String(repeating: "\u{3042}", count: 250), size: 100, overlap: 20)
        #expect(pieces.count == 3)
        #expect(pieces[0].count == 100)
    }

    @Test("an empty document yields no passages")
    func emptySplit() {
        #expect(RAGContext.chunks("").isEmpty)
    }

    @Test("packing never exceeds the token budget and keeps the best first")
    func packStaysUnderBudget() {
        let sources = (1...12).map {
            RAGSource(title: "\u{51FA}\u{5178}\($0)", url: "https://example.com/\($0)", date: "2026-09-0\($0 % 9 + 1)",
                      text: String(repeating: "\u{672C}\u{6587}", count: 100))
        }
        let packed = RAGContext.pack(sources, budget: 900, count: fakeCount)
        #expect(!packed.isEmpty)
        #expect(packed.count < sources.count)
        #expect(packed.first == sources.first)
        let used = packed.map { fakeCount(RAGContext.measuredBlock($0)) + 2 }.reduce(0, +)
        #expect(used <= 900)
    }

    @Test("a single passage larger than the budget is dropped, not truncated into the prompt")
    func oversizedPassageDropped() {
        let huge = RAGSource(title: "\u{9577}\u{3044}", url: "https://example.com/a", text: String(repeating: "\u{5B57}", count: 5_000))
        let small = RAGSource(title: "\u{77ED}\u{3044}", url: "https://example.com/b", text: "\u{77ED}\u{3044}\u{672C}\u{6587}")
        let packed = RAGContext.pack([huge, small], budget: 900, count: fakeCount)
        #expect(packed == [small])
    }

    @Test("the prompt numbers the sources and the citations match those numbers")
    func promptAndCitations() {
        let sources = [
            RAGSource(title: "\u{70B9}\u{691C}\u{624B}\u{9806}\u{66F8}", url: "tenken.md", date: "2026-04-01", text: "\u{6708}\u{6B21}\u{70B9}\u{691C}\u{306F}\u{6BCE}\u{6708}1\u{65E5}\u{306B}\u{884C}\u{3046}\u{3002}"),
            RAGSource(title: "\u{30CB}\u{30E5}\u{30FC}\u{30B9}", url: "https://example.com/n", text: "\u{672C}\u{6587}")
        ]
        let prompt = RAGContext.prompt(question: "\u{6708}\u{6B21}\u{70B9}\u{691C}\u{306F}\u{3044}\u{3064}?", sources: sources, today: "2026-09-24")
        #expect(prompt.hasPrefix(RAGContext.instruction(today: "2026-09-24")))
        #expect(prompt == RAGContext.instruction(today: "2026-09-24") + "\n\n" + RAGContext.message(question: "\u{6708}\u{6B21}\u{70B9}\u{691C}\u{306F}\u{3044}\u{3064}?", sources: sources))
        #expect(prompt.contains("\u{8CC7}\u{6599}[1] \u{70B9}\u{691C}\u{624B}\u{9806}\u{66F8} / tenken.md / 2026-04-01"))
        #expect(prompt.contains("\u{8CC7}\u{6599}[2] \u{30CB}\u{30E5}\u{30FC}\u{30B9} / https://example.com/n / \u{65E5}\u{4ED8}\u{4E0D}\u{660E}"))
        #expect(prompt.hasSuffix("\u{8CEA}\u{554F}: \u{6708}\u{6B21}\u{70B9}\u{691C}\u{306F}\u{3044}\u{3064}?"))

        let citations = RAGContext.citations(sources)
        #expect(citations == [
            "[1] \u{70B9}\u{691C}\u{624B}\u{9806}\u{66F8} \u{2014} tenken.md (2026-04-01)",
            "[2] \u{30CB}\u{30E5}\u{30FC}\u{30B9} \u{2014} https://example.com/n (\u{65E5}\u{4ED8}\u{4E0D}\u{660E})"
        ])
    }

    @Test("the instruction is byte-identical between turns so the KV prefix is reused")
    func instructionIsStable() {
        let a = RAGContext.prompt(question: "\u{8CEA}\u{554F}A", sources: [], today: "2026-09-24")
        let b = RAGContext.prompt(question: "\u{307E}\u{3063}\u{305F}\u{304F}\u{5225}\u{306E}\u{8CEA}\u{554F}B", sources: [], today: "2026-09-24")
        let shared = zip(a, b).prefix { $0 == $1 }.count
        #expect(shared >= RAGContext.instruction(today: "2026-09-24").count)
    }

    @Test("two Japanese documents are indexed and the right one comes back")
    func localDocumentsSearch() throws {
        let directory = URL(filePath: NSTemporaryDirectory())
            .appending(path: "rag-test-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        try Data("""
            # \u{767A}\u{96FB}\u{6A5F} \u{70B9}\u{691C}\u{624B}\u{9806}\u{66F8} 2026-04-01
            \u{6708}\u{6B21}\u{70B9}\u{691C}\u{3067}\u{306F}\u{51B7}\u{5374}\u{6C34}\u{306E}\u{91CF}\u{3068}\u{6CB9}\u{5727}\u{3092}\u{78BA}\u{8A8D}\u{3059}\u{308B}\u{3002}\u{51B7}\u{5374}\u{6C34}\u{304C}\u{4E0B}\u{9650}\u{7DDA}\u{3092}\u{4E0B}\u{56DE}\u{3063}\u{3066}\u{3044}\u{305F}\u{3089}\u{88DC}\u{5145}\u{3059}\u{308B}\u{3002}
            """.utf8).write(to: directory.appending(path: "tenken.md"))
        try Data("""
            # \u{793E}\u{5185}\u{30CD}\u{30C3}\u{30C8}\u{30EF}\u{30FC}\u{30AF} \u{5229}\u{7528}\u{898F}\u{7D04}
            \u{7121}\u{7DDA}LAN\u{306E}\u{30D1}\u{30B9}\u{30EF}\u{30FC}\u{30C9}\u{306F}\u{534A}\u{5E74}\u{3054}\u{3068}\u{306B}\u{5909}\u{66F4}\u{3059}\u{308B}\u{3002}\u{6765}\u{5BA2}\u{7528}\u{30CD}\u{30C3}\u{30C8}\u{30EF}\u{30FC}\u{30AF}\u{306F}\u{696D}\u{52D9}\u{7AEF}\u{672B}\u{306B}\u{4F7F}\u{308F}\u{306A}\u{3044}\u{3002}
            """.utf8).write(to: directory.appending(path: "network.txt"))
        try Data("noise".utf8).write(to: directory.appending(path: "notes.json"))

        let passages = RAGDocuments.load(directory: directory)
        #expect(passages.count == 2)
        #expect(Set(passages.map(\.url)) == ["tenken.md", "network.txt"])
        #expect(passages.first { $0.url == "tenken.md" }?.date == "2026-04-01")
        #expect(passages.first { $0.url == "network.txt" }?.date == "")

        let selected = RAGContext.select(passages, question: "\u{51B7}\u{5374}\u{6C34}\u{306F}\u{3069}\u{3046}\u{3059}\u{308B}?", count: fakeCount)
        #expect(selected.first?.url == "tenken.md")
        #expect(selected.first?.text.contains("\u{51B7}\u{5374}\u{6C34}") == true)
    }

    @Test("a question that matches nothing still returns passages, just unranked")
    func rankingFallsThrough() {
        let passages = [RAGSource(title: "a", url: "a", text: "\u{307E}\u{3063}\u{305F}\u{304F}\u{95A2}\u{4FC2}\u{306E}\u{306A}\u{3044}\u{672C}\u{6587}")]
        #expect(RAGContext.rank(passages.map(\.text), query: "xyzzy").isEmpty)
        #expect(RAGContext.select(passages, question: "xyzzy", count: fakeCount) == passages)
    }

    @Test("the published date comes from the meta tag, the time tag, then the text")
    func dateExtraction() {
        let meta = """
            <html><head><meta charset="utf-8">
            <meta property="article:published_time" content="2026-09-22T08:15:00+09:00">
            <time datetime="2020-01-01">ignored</time></head><body>2019/05/05</body></html>
            """
        #expect(RAGContext.publishedDate(html: meta) == "2026-09-22")

        let time = "<article><time datetime='2026-09-23T10:00:00Z'>\u{304D}\u{306E}\u{3046}</time><p>\u{672C}\u{6587}</p></article>"
        #expect(RAGContext.publishedDate(html: time) == "2026-09-23")

        let plain = "<div class='byline'>2026\u{5E74}9\u{6708}7\u{65E5} \u{914D}\u{4FE1}</div><p>\u{672C}\u{6587}</p>"
        #expect(RAGContext.publishedDate(html: plain) == nil)
        #expect(RAGContext.isoDate(in: "2026\u{5E74}9\u{6708}7\u{65E5} \u{914D}\u{4FE1} \u{672C}\u{6587}") == "2026-09-07")

        #expect(RAGContext.publishedDate(html: "<p>\u{65E5}\u{4ED8}\u{306E}\u{306A}\u{3044}\u{672C}\u{6587}</p>") == nil)
        #expect(RAGContext.publishedDate(html: "<meta name='author' content='2026-01-01'>") == nil)
    }

    @Test("dates in search snippets are normalised")
    func snippetDates() {
        #expect(RAGContext.isoDate(in: "2026/9/5 \u{2014} \u{8A18}\u{4E8B}\u{306E}\u{8981}\u{7D04}") == "2026-09-05")
        #expect(RAGContext.isoDate(in: "2026\u{5E74}12\u{6708}31\u{65E5}") == "2026-12-31")
        #expect(RAGContext.isoDate(in: "2026-13-01") == nil)
        #expect(RAGContext.isoDate(in: "no date here") == nil)
    }
}
