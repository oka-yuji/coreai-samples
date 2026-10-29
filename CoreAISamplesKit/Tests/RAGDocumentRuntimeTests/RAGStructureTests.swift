import Foundation
import RAGContext
import RAGDocumentRuntime
import Testing

@Suite("Original document structure")
struct RAGStructureTests {
    private func document(_ raw: String, format: RAGDocumentFormat, id: String = "document", text: String? = nil)
        -> RAGDocumentSnapshot
    {
        RAGDocumentSnapshot(
            id: id, url: "document", finalURL: "document", title: "Fixture", raw: raw,
            text: text ?? (format == .html ? RAGHTMLText.plainText(fromHTML: raw) : raw), format: format)
    }

    @Test(
        "RG-D01 keeps parent headings and distinct repeated section names",
        arguments: [RAGDocumentFormat.html, .markdown])
    func hierarchy(_ format: RAGDocumentFormat) throws {
        let raw =
            format == .html
            ? "<h1>Manual</h1><h2>Server</h2><h3>Reconnect</h3><p>Wait 120 seconds.</p><h2>Client</h2><h3>Reconnect</h3><p>Wait 20 seconds.</p>"
            : "# Manual\n## Server\n### Reconnect\nWait 120 seconds.\n## Client\n### Reconnect\nWait 20 seconds.\n"
        let original = document(raw, format: format)
        let parsed = RAGStructureReader.read(original)
        let structure = try #require(parsed.structure)
        #expect(structure.isAvailable, "\(structure.unavailableReason ?? "unknown")")
        #expect(structure.sections.map(\.title) == ["Manual", "Server", "Reconnect", "Client", "Reconnect"])
        let sections = structure.sections
        try #require(sections.count == 5)
        #expect(sections[2].parentID == sections[1].id)
        #expect(sections[4].parentID == sections[3].id)
        #expect(Set(sections.map(\.id)).count == 5)
        #expect(
            sections[1].body.text(in: original.text)?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == true)
        #expect(sections[2].body.text(in: original.text)?.contains("120") == true)
        #expect(sections[2].body.text(in: original.text)?.contains("Wait 20 seconds") == false)
        #expect(parsed.text.utf8.elementsEqual(original.text.utf8))
        #expect(parsed.version == original.version)
    }

    @Test("RG-D02 preserves exact Unicode ranges and UTF8 source columns")
    func unicodeRanges() throws {
        let raw = "# Cafe\u{301} \u{1F469}\u{200D}\u{1F4BB}\r\nBody\r\nNext\r\n====\r\nEnd\r\n"
        let original = document(raw, format: .markdown)
        let structure = try #require(RAGStructureReader.read(original).structure)
        #expect(structure.isAvailable, "\(structure.unavailableReason ?? "unknown")")
        try #require(structure.sections.count == 2)
        #expect(structure.sections[0].heading.text(in: raw)?.contains("Cafe\u{301}") == true)
        #expect(structure.sections[1].heading.text(in: raw)?.contains("====") == true)
        #expect(
            structure.sections[1].body.text(in: raw)?.contains("End") == true,
            "sections=\(structure.sections); rawCount=\(raw.count)")
        let positions = RAGTextCoordinates("\u{65E5}\u{672C}\nX")
        #expect(positions.characterOffset(line: 1, column: 4) == 1)
        #expect(positions.characterOffset(line: 1, column: 2) == nil)
        #expect(positions.characterOffset(line: 2, column: 1) == 3)
        #expect(positions.characterOffset(line: 1, column: Int.max) == nil)
    }

    @Test("RG-D03 does not promote code fences, indented code or raw HTML to headings")
    func markdownBoundaries() throws {
        let raw =
            "# Real\n````swift\n## Fake\n```\n## Still fake\n````\n\n    ## Indented\n\n<script>\n# Script\n</script>\n\nTitle\nline two\n---\nBody\n"
        let structure = try #require(RAGStructureReader.read(document(raw, format: .markdown)).structure)
        #expect(structure.isAvailable)
        #expect(structure.sections.count == 2)
        #expect(structure.sections.first?.title == "Real")
        #expect(structure.sections.last?.title.contains("line two") == true)
    }

    @Test("RG-D03 excludes non-body headings and preserves encoded inline content")
    func htmlBoundaries() throws {
        let raw =
            "<html><head><style>h1{color:red}</style></head><body><nav><h2>Menu</h2></nav><main><h1>Manual</h1><h3>Mac &amp; <em>iPhone</em></h3><p>Body &#x65E5;.</p><script>'<h2>Fake</h2>'</script></main><footer>Copyright</footer></body></html>"
        let original = document(raw, format: .html)
        let structure = try #require(RAGStructureReader.read(original).structure)
        #expect(structure.isAvailable, "\(structure.unavailableReason ?? "unknown")")
        #expect(structure.sections.map(\.title) == ["Manual", "Mac & iPhone"])
        #expect(structure.sections.last?.body.text(in: original.text)?.contains("Copyright") == false)
    }

    @Test("RG-D04 keeps text when structure is unsupported, hidden or inconsistent")
    func fallbacks() throws {
        for format in [RAGDocumentFormat.text, .pdfText, .renderedText] {
            let original = document("Readable text", format: format)
            let result = RAGStructureReader.read(original)
            #expect(result.structure?.unavailableReason == "unsupportedFormat")
            #expect(result.text == original.text)
        }
        let hidden = RAGStructureReader.read(
            document("<main><h1>Title</h1><div hidden><h2>Hidden</h2></div></main>", format: .html))
        #expect(hidden.structure?.isAvailable == false)
        let changed = document("<h1>Title</h1>", format: .html, text: "Other text")
        #expect(RAGStructureReader.read(changed).structure?.unavailableReason == "normalizationMismatch")
    }

    @Test("Projection rejects marker collisions and boundaries inside a grapheme")
    func projection() throws {
        let projected = try #require(
            RAGTextProjection.removingMarkers(from: "A  XX0; \n BXX1;", prefix: "XX", markers: ["XX0;", "XX1;"]))
        #expect(projected.text == "A\nB")
        #expect(projected.markerEnds == ["XX0;": 1, "XX1;": 3])
        #expect(RAGTextProjection.removingMarkers(from: "XX0;XX0;", prefix: "XX", markers: ["XX0;"]) == nil)
        #expect(RAGTextProjection.removingMarkers(from: "eXX0;\u{301}", prefix: "XX", markers: ["XX0;"]) == nil)
    }

    @Test(
        "Hidden and non-body ancestors cannot become the selected article",
        arguments: [
            "nav", "aside", "footer", "template", "div hidden", "div aria-hidden='true'", "div style='display:none'",
            "div style='visibility:hidden'",
        ])
    func excludedAncestors(_ tag: String) {
        let close = tag.split(separator: " ").first.map(String.init) ?? "div"
        let html = "<\(tag)><article><h1>Hidden</h1><p>Hidden body</p></article></\(close)>"
        let result = RAGStructureReader.read(document(html, format: .html))
        #expect(result.structure?.isAvailable == false)
    }

    @Test("Content and preamble ranges separate relevant introduction from surrounding boilerplate")
    func preamble() throws {
        let html = "<nav>Menu</nav><main><p>Introduction</p><h1>Title</h1><p>Body</p></main><footer>Footer</footer>"
        let original = document(html, format: .html)
        let result = try #require(RAGStructureReader.read(original).structure)
        #expect(result.isAvailable)
        #expect(result.content?.text(in: original.text) == "Introduction\nTitle\nBody")
        #expect(
            result.preamble?.text(in: original.text)?.trimmingCharacters(in: .whitespacesAndNewlines) == "Introduction")
    }

    @Test("Encoded snapshots carry the source hash and reject mismatched decoded versions")
    func snapshotVersion() throws {
        let original = document("# Title\nBody", format: .markdown)
        let data = try JSONEncoder().encode(RAGStructureReader.read(original))
        let decoded = try JSONDecoder().decode(RAGDocumentSnapshot.self, from: data)
        #expect(decoded.version == RAGContext.fingerprint(original.text))
        let json = String(decoding: data, as: UTF8.self)
        #expect(json.contains("\"version\""))
        let changed = json.replacingOccurrences(of: original.version, with: String(repeating: "0", count: 64))
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(RAGDocumentSnapshot.self, from: Data(changed.utf8))
        }
    }

    @Test("Section identities include the document, version and source position")
    func identities() throws {
        let first = document("# Title\nBody", format: .markdown)
        let other = document(first.raw, format: .markdown, id: "another document")
        let a = try #require(RAGStructureReader.read(first).structure?.sections.first)
        let b = try #require(RAGStructureReader.read(other).structure?.sections.first)
        #expect(a.id != b.id)
        let changed = document(first.raw + " changed", format: .markdown)
        let c = try #require(RAGStructureReader.read(changed).structure?.sections.first)
        #expect(a.id != c.id)
    }
}
