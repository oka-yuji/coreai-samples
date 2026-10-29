import Foundation
import Testing

@testable import ConversationFeature

@Suite("Source group packing")
struct SourceGroupTests {
    private func source(_ text: String, offset: Int) -> ConversationSource {
        ConversationSource(
            document: "manual", version: String(repeating: "a", count: 64), title: "Manual",
            passages: [.init(offset: offset, text: text)])
    }

    @Test("Final prompt packing cannot retain a heading after dropping its body")
    func atomicBudget() throws {
        let ledger = ConversationSources()
        let header = source("Title", offset: 0)
        let body = source(String(repeating: "x", count: 500), offset: 6)
        let headerOnly = try #require(
            try ledger.prepare(
                question: "Question", sources: [header], today: "today",
                budget: 10_000, count: { $0.count }))
        let ordinary = try ledger.prepare(
            question: "Question", sources: [header, body], today: "today",
            budget: headerOnly.tokens, count: { $0.count })
        let atomic = try ledger.prepare(
            question: "Question", groups: [[header, body]], today: "today",
            budget: headerOnly.tokens, count: { $0.count })
        #expect(ordinary?.sources == [header])
        #expect(atomic == nil)
    }

    @Test("Shared headings are counted once and complete groups survive archive round trips")
    func sharedHeadingAndArchive() throws {
        var ledger = ConversationSources()
        let header = source("Title", offset: 0)
        let first = source("First body", offset: 6)
        let second = source("Second body", offset: 17)
        let one = try #require(
            try ledger.prepare(
                question: "Question", groups: [[header, first]], today: "today",
                budget: 10_000, count: { $0.count }))
        let packed = try #require(
            try ledger.prepare(
                question: "Question", groups: [[header, first], [header, second]],
                today: "today", budget: one.tokens, count: { $0.count }))
        #expect(packed.groupCount == 1)
        #expect(packed.sources == [header, first])
        #expect(try ledger.register(packed.sources) == ["S1"])
        let restored = try JSONDecoder().decode(ConversationSources.self, from: JSONEncoder().encode(ledger))
        #expect(restored == ledger)
        #expect(restored.records.first?.passages == header.passages + first.passages)
    }
}
