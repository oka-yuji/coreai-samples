import Foundation
import RAGContext
import Testing

struct DocumentSourcesTests {
    @Test("Passage offsets identify exact characters in a stable document version")
    func exactPassages() {
        let text = String(repeating: "Alpha 👩‍💻 beta\n", count: 100)
        let sources = RAGContext.documentSources(title: "Manual", url: "file:///manual", text: text)
        #expect(sources.count > 1)
        #expect(Set(sources.compactMap(\.version)).count == 1 && sources.first?.version?.count == 64)
        for source in sources {
            #expect(source.text == String(text.dropFirst(source.offset ?? 0).prefix(source.text.count)))
            #expect(source.document == "file:///manual")
        }
        let again = RAGContext.documentSources(title: "Manual", url: "file:///manual", text: text)
        #expect(again == sources)
        let changed = RAGContext.documentSources(title: "Manual", url: "file:///manual", text: text + "New")
        #expect(changed.first?.version != sources.first?.version)
    }
}
