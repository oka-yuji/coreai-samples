import Foundation
import Testing

@testable import RAGContext

private enum LegacyHTMLText {
    static func plainText(fromHTML html: String) -> String {
        var text = html.replacing(/<!--.*?-->/.dotMatchesNewlines(), with: " ")
        text = text.replacing(/<script\b.*?<\/script>/.dotMatchesNewlines().ignoresCase(), with: " ")
        text = text.replacing(/<style\b.*?<\/style>/.dotMatchesNewlines().ignoresCase(), with: " ")
        text = text.replacing(/<noscript\b.*?<\/noscript>/.dotMatchesNewlines().ignoresCase(), with: " ")
        text = text.replacing(/<(br|\/p|\/div|\/h[1-6]|\/li|\/tr|\/table)\b[^>]*>/.ignoresCase(), with: "\n")
        text = text.replacing(/<[^>]*>/.dotMatchesNewlines(), with: " ")
        return RAGHTMLText.condense(RAGHTMLText.decodeEntities(text))
    }

    private static let metaNames = [
        "article:published_time", "article:published", "og:published_time",
        "datePublished", "pubdate", "date", "dc.date", "dc.date.issued",
    ]

    static func publishedDate(html: String) -> String? {
        for tag in html.matches(of: /<meta\b[^>]*>/.ignoresCase()) {
            let attributes = String(tag.0)
            guard
                let name = RAGContext.attribute("property", in: attributes)
                    ?? RAGContext.attribute("name", in: attributes)
                    ?? RAGContext.attribute("itemprop", in: attributes),
                metaNames.contains(where: { $0.caseInsensitiveCompare(name) == .orderedSame }),
                let content = RAGContext.attribute("content", in: attributes),
                let date = RAGContext.isoDate(in: content)
            else { continue }
            return date
        }
        for tag in html.matches(of: /<time\b[^>]*>/.ignoresCase()) {
            guard let value = RAGContext.attribute("datetime", in: String(tag.0)),
                let date = RAGContext.isoDate(in: value)
            else { continue }
            return date
        }
        return nil
    }
}

private struct SplitMix64 {
    var state: UInt64

    mutating func next(below bound: Int) -> Int {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return Int((value ^ (value >> 31)) % UInt64(bound))
    }
}

private let fragments = [
    "<", ">", "<!--", "-->", "--", "-", "!", "<!-", "</", "<p>", "<b>", "</td>", "<a href='x'>",
    "<script", "<SCRIPT", "<Script ", "<scripts", "<script.x", "<script'x", "<script_x", "<script:x>", "<script1>",
    "<script>", "</script>", "</SCRIPT>", "</SCRIPT >", "</script", "<\u{17F}cript", "</\u{17F}cript>", "<scr\u{130}pt",
    "<style", "<style>", "</style>", "<STYLE type=x>", "</Style>", "<noscript", "<noscript>", "</noscript>",
    "</NoScript>",
    "<br", "<br>", "<BR/>", "<BR\t/>", "<br\u{AD}>", "</p>", "</P>", "</p", "</pre>", "</div>", "</DIV >",
    "</h1>", "</h6>", "</H6\r\n>", "</h7>", "</h1x>", "</h1\u{301}>", "</li>", "</li's>", "</tr>", "</tr.x>",
    "</table>", "</table\u{200D}>",
    "<meta property=\"article:published_time\" content=\"2026-09-22\">", "<time datetime=\"2026-09-23\">",
    "<meta name='date' content='2026/9/5'>", "<meta name=\"author\" content=\"2026-01-01\">",
    "<META ITEMPROP=\"datePublished\" CONTENT=\"2026\u{5E74}9\u{6708}7\u{65E5}\">",
    "<metadata property=\"article:published_time\" content=\"2026-01-02\">", "<times datetime=\"2026-05-06\">",
    "<meta_x name='date' content='2026-07-08'>", "<Time datetime='2026-08-09'>",
    "<meta", "<time", "<metadata>", "<meta.x", "<times>", "<time datetime='x'>",
    " content=\"2027-02-03\"", " property='og:published_time'", " datetime=\"2025-12-31\"", "\"", "'", "=",
    "&amp;", "&#x41;", "&#0;", "&nbsp;", "&lt;", "&", ";",
    "\n", "\r\n", "\r", "\t", " ", "a", "x", "1", ".", "_",
    "\u{17F}", "\u{212A}", "\u{130}", "\u{301}", "\u{338}", "\u{AD}", "\u{200D}", "\u{600}", "<\u{301}",
    "\u{1F600}", "\u{1F469}\u{200D}", "\u{1F1EF}\u{1F1F5}", "\u{65E5}\u{672C}\u{8A9E}",
]

private let generatedHTML: [String] = {
    var generator = SplitMix64(state: 392)
    return (0..<2_000).map { _ in
        (0...generator.next(below: 48)).map { _ in fragments[generator.next(below: fragments.count)] }.joined()
    }
}()

private let fixedHTML = [
    "",
    """
    <!DOCTYPE html><html><head><meta charset="utf-8"><title>Manual</title>
    <style>body { color: red; }</style><script>var a = "</p>"; if (a < b) {}</script></head>
    <body><h1>Title</h1><p>Hello&nbsp;<b>world</b> &amp; friends&hellip;</p><br/><ul><li>One</li><li>Two</li></ul>
    <table><tr><td>A</td><td>B</td></tr></table><noscript>JS off</noscript><!-- note --></body></html>
    """,
    """
    <html><head><meta charset="utf-8">
    <meta property="article:published_time" content="2026-09-22T08:15:00+09:00">
    <time datetime="2020-01-01">ignored</time></head><body>2019/05/05</body></html>
    """,
    "<article><time datetime='2026-09-23T10:00:00Z'>\u{304D}\u{306E}\u{3046}</time><p>\u{672C}\u{6587}</p></article>",
    "<div class='byline'>2026\u{5E74}9\u{6708}7\u{65E5} \u{914D}\u{4FE1}</div><p>\u{672C}\u{6587}</p>",
    "<meta name='author' content='2026-01-01'>",
    "<META NAME=\"DATE\" CONTENT=\"2026/9/5\"><meta name=date content=2026-01-02>",
    "<meta property='og:published_time' content='2026\u{5E74}9\u{6708}7\u{65E5}'><time datetime=\"2026-13-01\">",
    "<meta property=\"article:published_time\" content=\"2026-09-22\"<time datetime=\"2026-09-23\">",
    "<meta<time datetime='2026-10-30'>",
    "<metadata property=\"article:published_time\" content=\"2026-01-02\"><times datetime=\"2026-05-06\">"
        + "<meta.x name='date' content='2026-07-08'><time datetime=\"2026-03-04\">",
    "<!-- open comment <p>kept</p>",
    "<!-- a <!-- b --> c -->d",
    "<!--<script>-->x</script>y",
    "<!---->a<!--->b-->c<!-->d-->",
    "<script>never closed <p>text</p>",
    "<style>x<noscript>y",
    "<SCRIPT type=x>a</Script >b</SCRIPT>c",
    "<scripts>x</scripts><script.x>y</script><script'x>z</script><script_x>w</script>v",
    "<\u{17F}cript>s</script>t<script>u</\u{17F}cript>v</script>",
    "a<br>b<BR/>c</P>d</p >e</pre>f</h1>g</h7>h</h1x>i</li>j</tr>k</table>l</div>m",
    "<br",
    "text < more",
    "a > b < c > d",
    "<p>\u{65E5}\u{672C}\u{8A9E}</p>\r\n<p>\u{1F600} \u{1F1EF}\u{1F1F5}</p>",
    "<\u{301}b>x</b\u{301}>y>\u{301}z<p>w\u{338}><i>\u{301}j>k",
    "\u{600}<b>x</b>\u{17F}\u{212A}\u{130}",
    "&#x41;&#0;&#65;&amp;lt;&unknown;&#x110000;&",
    "<h1>Manual</h1><h2>Server</h2><h3>Reconnect</h3><p>Wait 120 seconds.</p><h2>Client</h2><h3>Reconnect</h3>",
]

@Suite("Linear HTML normalisation")
struct RAGHTMLTextTests {
    @Test("HTML-01 fixed pages normalise to the same bytes as the legacy regex implementation", arguments: fixedHTML)
    func fixedPlainText(_ html: String) {
        let text = RAGHTMLText.plainText(fromHTML: html)
        let legacy = LegacyHTMLText.plainText(fromHTML: html)
        #expect(text.utf8.elementsEqual(legacy.utf8), "\(text.debugDescription) != \(legacy.debugDescription)")
    }

    @Test("HTML-01 seeded generated inputs normalise to the same bytes as the legacy regex implementation")
    func generatedPlainText() {
        #expect(generatedHTML.count >= 1_000)
        for html in generatedHTML {
            let text = RAGHTMLText.plainText(fromHTML: html)
            let legacy = LegacyHTMLText.plainText(fromHTML: html)
            #expect(
                text.utf8.elementsEqual(legacy.utf8),
                "\(html.debugDescription): \(text.debugDescription) != \(legacy.debugDescription)")
        }
    }

    @Test("HTML-02 fixed and generated pages yield the same published date as the legacy regex implementation")
    func publishedDates() {
        let pages = fixedHTML + generatedHTML
        let legacy = pages.map(LegacyHTMLText.publishedDate)
        #expect(legacy.contains(nil) && legacy.contains { $0 != nil })
        for (html, date) in zip(pages, legacy) {
            #expect(RAGContext.publishedDate(html: html) == date, "\(html.debugDescription)")
        }
    }

    @Test(
        "HTML-03 plain text and published date each finish within two seconds on 256 KiB of unclosed openers",
        .serialized,
        arguments: [
            "<", "<!--", "<script", "<style", "<noscript", "<br", "</p", "</div", "</h1", "</li", "</tr", "</table",
            "<meta", "<time",
        ])
    func unclosedOpeners(_ opener: String) {
        let html = String(repeating: opener, count: (262_144 + opener.utf8.count - 1) / opener.utf8.count)
        let clock = ContinuousClock()
        let text = clock.measure { _ = RAGHTMLText.plainText(fromHTML: html) }
        let date = clock.measure { _ = RAGContext.publishedDate(html: html) }
        #expect(text < .seconds(2), "plainText took \(text)")
        #expect(date < .seconds(2), "publishedDate took \(date)")
    }

    @Test(
        "HTML-05 published date finishes within two seconds on 256 KiB of closed meta and time tags",
        .serialized,
        arguments: [
            #"<meta name="date" content="x">"#, #"<meta property="og:title" content="x">"#, #"<time datetime="x">"#,
        ])
    func closedTags(_ tag: String) {
        let html = String(repeating: tag, count: (262_144 + tag.utf8.count - 1) / tag.utf8.count)
        let date = ContinuousClock().measure { #expect(RAGContext.publishedDate(html: html) == nil) }
        #expect(date < .seconds(2), "publishedDate took \(date)")
    }
}
