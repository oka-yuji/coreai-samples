import Foundation
import Testing

@testable import RAGContext

@Suite("Structured source selection")
struct RAGStructuredSelectionTests {
    private func fixture() -> RAGDocumentSnapshot {
        let text = "Manual\niPhone\nWait seven.\nMac\nWait thirty."
        var document = RAGDocumentSnapshot(
            id: "manual", url: "manual", finalURL: "manual", title: "Devices",
            raw: text, text: text, format: .markdown)
        let headings = ["Manual", "iPhone", "Mac"].enumerated().map { index, title in
            let range = text.range(of: title)
            let offset = range.map { text.distance(from: text.startIndex, to: $0.lowerBound) } ?? 0
            return RAGHeading(
                level: index == 0 ? 1 : 2, title: title, range: RAGTextRange(offset, offset + title.count))
        }
        document.structure = .make(documentID: document.id, text: text, headings: headings)
        return document
    }

    @Test("RG-B01 retrieves a body using its parent heading without fabricating evidence")
    func headingMatch() throws {
        let document = fixture()
        let candidates = RAGStructuredSelection.candidates(document).candidates
        let selected = RAGStructuredSelection.select(
            candidates, question: "Mac", budget: 200, count: { $0.count }, rank: RAGContext.rank)
        let first = try #require(selected.candidates.first)
        #expect(first.source.text.contains("thirty"))
        #expect(!first.source.text.contains("Mac"))
        #expect(first.headers.contains { $0.text == "Mac" })
        for source in selected.sources {
            let offset = try #require(source.offset)
            let range = RAGTextRange(offset, offset + source.text.count)
            #expect(range.text(in: document.text)?.utf8.elementsEqual(source.text.utf8) == true)
            #expect(source.version == document.version)
        }
    }

    @Test("RG-B02 accounts for headers atomically and deduplicates shared original ranges")
    func groupBudget() throws {
        let candidates = RAGStructuredSelection.candidates(fixture()).candidates
        let all = RAGStructuredSelection.select(
            candidates, question: "Wait", budget: 10_000, count: { $0.count }, rank: RAGContext.rank)
        let cost = all.sources.reduce(0) { $0 + RAGContext.measuredBlock($1).count + 2 }
        let packed = RAGStructuredSelection.select(
            candidates, question: "Wait", budget: cost, count: { $0.count }, rank: RAGContext.rank)
        #expect(packed.sources == all.sources)
        #expect(packed.sources.filter { $0.text == "Manual" }.count == 1)
        let none = RAGStructuredSelection.select(
            candidates, question: "Mac", budget: 1, count: { $0.count }, rank: RAGContext.rank)
        #expect(none.sources.isEmpty && none.hasMatches)
        #expect(
            RAGStructuredSelection.select(
                candidates, question: "Mac", budget: -1, count: { $0.count }, rank: RAGContext.rank
            ).sources.isEmpty)
    }

    @Test("RG-B02 splits long bodies within section boundaries with original document versions")
    func longBodies() throws {
        let raw = "First\n" + String(repeating: "a", count: 1_200) + "\nSecond\n" + String(repeating: "b", count: 700)
        var document = RAGDocumentSnapshot(
            id: "long", url: "long", finalURL: "long", title: "Long", raw: raw, text: raw, format: .markdown)
        let boundary = 1_207
        document.structure = .make(
            documentID: "long", text: raw,
            headings: [
                .init(level: 1, title: "First", range: .init(0, 5)),
                .init(level: 1, title: "Second", range: .init(boundary, boundary + 6)),
            ])
        let candidates = RAGStructuredSelection.candidates(document).candidates
        #expect(candidates.count > 3)
        for candidate in candidates {
            let offset = try #require(candidate.source.offset)
            #expect(candidate.source.text.count <= 500)
            #expect(candidate.source.version == document.version)
            #expect(!(offset < boundary && offset + candidate.source.text.count > boundary))
        }
    }

    @Test("RG-B03 preserves flat fallback reasons and separates identical content at different URLs")
    func fallbackIdentity() {
        let raw = "Flat text."
        let first = RAGDocumentSnapshot(
            id: "first", url: "first", finalURL: "first", title: "Text", raw: raw, text: raw, format: .text)
        let second = RAGDocumentSnapshot(
            id: "second", url: "second", finalURL: "second", title: "Text", raw: raw, text: raw, format: .text)
        let a = RAGStructuredSelection.candidates(first)
        let b = RAGStructuredSelection.candidates(second)
        #expect(a.fallbackReason == "structureNotCaptured")
        #expect(a.candidates.first?.sectionID != b.candidates.first?.sectionID)
        #expect(a.candidates.map(\.source) == first.sources)
    }

    @Test("Ranking without matches retains the baseline fall-through order")
    func unmatchedOrder() {
        let candidates = RAGStructuredSelection.candidates(fixture()).candidates
        let selection = RAGStructuredSelection.select(
            candidates, question: "absent", budget: 10_000, count: { $0.count }, rank: { _, _ in [] })
        #expect(!selection.hasMatches)
        #expect(selection.candidates.map(\.sectionID) == candidates.map(\.sectionID))
    }

    @Test("Duplicate candidates cannot consume multiple body ranks")
    func duplicateBodies() throws {
        let candidate = try #require(RAGStructuredSelection.candidates(fixture()).candidates.first)
        let selected = RAGStructuredSelection.select(
            [candidate, candidate, candidate], question: "Wait", budget: 10_000,
            count: { $0.count }, rank: { _, _ in [0, 1, 2] })
        #expect(selected.candidates.count == 1)
    }

    @Test("A heading-only leaf preserves its evidence through a documented flat fallback")
    func headingOnlyLeaf() {
        let text = "Manual\nFact only in heading"
        var document = RAGDocumentSnapshot(
            id: "leaf", url: "leaf", finalURL: "leaf", title: "Leaf", raw: text, text: text, format: .markdown)
        document.structure = .make(
            documentID: "leaf", text: text,
            headings: [
                .init(level: 1, title: "Manual", range: .init(0, 6)),
                .init(level: 2, title: "Fact only in heading", range: .init(7, text.count)),
            ])
        let set = RAGStructuredSelection.candidates(document)
        #expect(set.fallbackReason == "headingOnlyLeaf")
        #expect(set.candidates.map(\.source) == document.sources)
    }
}
