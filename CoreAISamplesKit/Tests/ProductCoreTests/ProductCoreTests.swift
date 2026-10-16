import Foundation
import ProductCore
import Testing

struct ProductCoreTests {
    @Test("Readable page text keeps its title and source while excluding scripts and navigation")
    func readablePage() throws {
        let html = """
            <title>Example &amp; specifications</title><nav>Other products</nav>
            <main><h2>Memory</h2><p>Up to 32&nbsp;GB</p><script>Fake 999 GB</script>
            <p>From &#x24;999</p><footer>Old products</footer></main>
            """
        let url = try #require(URL(string: "https://www.apple.com/example"))
        let stamp = Date(timeIntervalSince1970: 0)
        let source = try ProductPage.source(html: html, url: url, capturedAt: stamp)
        #expect(source.title == "Example & specifications")
        #expect(source.text == "Memory\nUp to 32 GB\nFrom $999")
        #expect(source.url == url)
        #expect(source.capturedAt == stamp)
    }

    @Test(
        "Only official HTTPS addresses without credentials or custom ports are accepted",
        arguments: [
            "http://www.apple.com/", "https://apple.com.example.org/", "https://www.microsoft.com@127.0.0.1/",
            "https://support.microsoft.com:444/", "https://user:pass@www.apple.com/", "file:///tmp/page",
        ])
    func invalidAddresses(value: String) {
        #expect(throws: ProductError.unsupportedURL) { try ProductPage.validatedURL(value) }
    }

    @Test("Supported official source URLs retain their path and query")
    func validAddress() throws {
        let value = "https://support.microsoft.com/ja-jp/surface/models/example?q=one"
        #expect(try ProductPage.validatedURL(value).absoluteString == value)
    }

    @Test("Empty and oversized descriptions fail instead of being silently shortened")
    func inputLimits() {
        #expect(throws: ProductError.emptyText) {
            try ProductSource(title: "", text: " \n ", url: nil, capturedAt: Date(timeIntervalSince1970: 0))
        }
        #expect(throws: ProductError.textTooLong(12_001)) {
            try ProductSource(
                title: "", text: String(repeating: "x", count: 12_001), url: nil,
                capturedAt: Date(timeIntervalSince1970: 0))
        }
    }

    @Test("Candidate evidence preserves qualifiers and separate configuration values")
    func preservesConditions() throws {
        let source = try ProductSource(
            title: "Example", text: "Memory\nUp to 32 GB\nStorage\n256 GB or 512 GB", url: nil,
            capturedAt: Date(timeIntervalSince1970: 0))
        let memo = try ProductMemo(
            source: source,
            candidates: [
                ProductCandidate(field: .memory, range: 13..<18, text: "32 GB", probability: 0.8),
                ProductCandidate(field: .memory, range: 13..<18, text: "32 GB", probability: 0.9),
                ProductCandidate(field: .storage, range: 27..<33, text: "256 GB", probability: 0.8),
                ProductCandidate(field: .storage, range: 37..<43, text: "512 GB", probability: 0.8),
            ])
        #expect(memo.values(for: .memory).count == 1)
        #expect(memo.values(for: .memory).first?.context == source.text)
        #expect(memo.values(for: .storage).count == 2)
        #expect(memo.values(for: .price).isEmpty)
    }

    @Test("Invalid offsets or fabricated candidate text are rejected")
    func rejectsInvalidEvidence() throws {
        let source = try ProductSource(
            title: "Example", text: "Memory 16 GB", url: nil, capturedAt: Date(timeIntervalSince1970: 0))
        #expect(throws: ProductError.invalidEvidence) {
            try ProductMemo(
                source: source,
                candidates: [
                    ProductCandidate(field: .memory, range: 7..<12, text: "32 GB", probability: 0.9)
                ])
        }
    }

    @Test("Chunk ranges cover Unicode scalars and make progress with overlap")
    func scalarChunks() throws {
        let text = "a\u{1F9ED}b\n" + String(repeating: "abc ", count: 300)
        let scalars = Array(text.unicodeScalars)
        var covered = Set<Int>()
        var start = 0
        while let chunk = ProductChunks.next(in: text, from: start, limit: 100) {
            #expect(chunk.text == String(String.UnicodeScalarView(scalars[chunk.range])))
            covered.formUnion(chunk.range)
            if chunk.range.upperBound == scalars.count { break }
            let next = ProductChunks.followingStart(after: chunk)
            #expect(next > start && next < chunk.range.upperBound)
            start = next
        }
        #expect(covered.count == scalars.count)
    }
    @Test("Hidden main and title markup never replaces visible page content")
    func ignoresHiddenMain() throws {
        let html = """
            <script>let template = "<title>Fake</title><main>Fake 999 GB</main>";</script>
            <!--<main>Invisible 512 GB</main>--><title>Visible product</title>
            <main><h2>Memory</h2><p>16 GB</p></main>
            """
        let url = try #require(URL(string: "https://www.apple.com/specs"))
        let source = try ProductPage.source(html: html, url: url, capturedAt: Date(timeIntervalSince1970: 0))
        #expect(source.title == "Visible product")
        #expect(source.text == "Memory\n16 GB")
    }

    @Test("Separate condition lines remain with the extracted value")
    func keepsMultiLineConditions() throws {
        let source = try ProductSource(
            title: "Example", text: "Battery\nUp to\n18 hours\nVideo playback only\n\nWeight\n1 kg", url: nil,
            capturedAt: Date(timeIntervalSince1970: 0))
        let memo = try ProductMemo(
            source: source,
            candidates: [
                ProductCandidate(field: .battery, range: 14..<22, text: "18 hours", probability: 0.8)
            ])
        #expect(memo.evidence.first?.context == "Battery\nUp to\n18 hours\nVideo playback only")
    }

    @Test("Canonically equivalent strings must still match the original scalar sequence")
    func exactSourceSequence() throws {
        let source = try ProductSource(
            title: "Example", text: "e\u{301}", url: nil, capturedAt: Date(timeIntervalSince1970: 0))
        #expect(throws: ProductError.invalidEvidence) {
            try ProductMemo(
                source: source,
                candidates: [ProductCandidate(field: .product, range: 0..<2, text: "\u{E9}", probability: 0.8)])
        }
    }

    @Test("Comment delimiters inside scripts cannot swallow visible document content")
    func commentInsideScript() throws {
        let html =
            #"<script>const start = "<!--";</script><title>Visible product</title><main>Memory 16 GB</main><!-- actual comment -->"#
        let source = try ProductPage.source(
            html: html, url: #require(URL(string: "https://www.apple.com/specs")),
            capturedAt: Date(timeIntervalSince1970: 0))
        #expect(source.title == "Visible product")
        #expect(source.text == "Memory 16 GB")
    }

    @Test("Nested inert templates are excluded before choosing the visible main element")
    func nestedTemplates() throws {
        let html = """
            <template><template><main>Hidden 999 GB</main></template><main>Hidden 512 GB</main></template>
            <!--<script>Comment script</script>--><title>Visible product</title><main>Memory 16 GB</main>
            """
        let source = try ProductPage.source(
            html: html, url: #require(URL(string: "https://www.apple.com/specs")),
            capturedAt: Date(timeIntervalSince1970: 0))
        #expect(source.title == "Visible product")
        #expect(source.text == "Memory 16 GB")
    }

}
