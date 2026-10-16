import ConversationFeature
import Foundation
import Testing

struct ConversationSourcesTests {
    func source(
        _ document: String = "file:///manual.md", _ version: String = "a", offset: Int = 0, text: String = "Alpha"
    ) -> ConversationSource {
        ConversationSource(
            document: document, version: String(repeating: version, count: 64), title: "Manual",
            passages: [SourcePassage(offset: offset, text: text)])
    }

    @Test("Source IDs stay unique and keep the document version and exact passage across round trips")
    func stableIDs() throws {
        var ledger = ConversationSources()
        #expect(try ledger.register([source(), source("file:///second.md")]) == ["S1", "S2"])
        #expect(try ledger.register([source(offset: 5, text: " beta")]) == ["S1"])
        #expect(
            ledger.records[0].passages == [
                SourcePassage(offset: 0, text: "Alpha"), SourcePassage(offset: 5, text: " beta"),
            ])
        #expect(try ledger.register([source()]) == ["S1"])
        #expect(ledger.records[0].passages.count == 2)
        #expect(try ledger.register([source("file:///manual.md", "b", text: "Revised")]) == ["S3"])
        let restored = try JSONDecoder().decode(ConversationSources.self, from: JSONEncoder().encode(ledger))
        try restored.validate()
        #expect(restored == ledger && restored.records[0].passages[0].text == "Alpha")
    }

    @Test("The expanded prompt fits the budget without registering discarded sources")
    func promptBudget() throws {
        let ledger = ConversationSources()
        let first = source()
        let second = source("file:///second.md", text: String(repeating: "More", count: 200))
        let complete = try #require(
            try ledger.prepare(
                question: "Question", sources: [first, second], today: "2026-10-16", budget: 10_000,
                count: { $0.count + 100 }))
        let kept = try #require(
            try ledger.prepare(
                question: "Question", sources: [first, second], today: "2026-10-16", budget: complete.tokens - 1,
                count: { $0.count + 100 }))
        #expect(kept.sources == [first] && kept.ids == ["S1"] && kept.tokens < complete.tokens)
        #expect(ledger.records.isEmpty)
        #expect(
            try ledger.prepare(
                question: "Question", sources: [first], today: "2026-10-16", budget: 1, count: { $0.count }) == nil)
    }

    @Test("A conflicting passage cannot redefine an existing source or consume an ID")
    func rejectsConflicts() throws {
        var ledger = ConversationSources()
        _ = try ledger.register([source()])
        let before = ledger
        #expect(throws: ConversationSources.Failure.self) {
            try ledger.register([source("file:///new.md"), source(offset: 1, text: "wrong")])
        }
        #expect(ledger == before)
        #expect(try ledger.register([source("file:///new.md")]) == ["S2"])
    }

    @Test("Citation checks distinguish unknown and missing IDs and ignore code examples")
    func sourceCitations() throws {
        var ledger = ConversationSources()
        _ = try ledger.register([source()])
        #expect(!ledger.check(answer: "Alpha [S1].", required: true).violation)
        #expect(ledger.check(answer: "Alpha [S2].", notes: ["[S9]"]).unknown == ["S2", "S9"])
        #expect(ledger.check(answer: "Alpha", required: true).missing)
        #expect(!ledger.check(answer: "Plain conversation").violation)
        #expect(!ledger.check(answer: "`[S8]`\n```\n[S9]\n```").violation)
        #expect(ledger.check(answer: "`Swift` [S9]").unknown == ["S9"])
        #expect(ledger.check(answer: "Known [S1].\n    ```\nUnknown [S99].", required: true).unknown == ["S99"])
    }

    @Test("A source question, its answer and a follow-up are saved in the same conversation")
    func sharesConversation() throws {
        var state = ConversationState(id: "sources", now: launch)
        let request = ConversationSourceRequest(web: false, local: true, budget: 3_000)
        let effects = ConversationFeature.reduce(&state, .sendSources("What is alpha?", request, now: launch))
        #expect(state.isRunning && state.isRetrieving)
        #expect(
            effects.contains {
                if case .run(.generate(_, _, let policy, _), _) = $0 {
                    return policy.sourceRequest == request && policy.allowedTools.isEmpty
                }
                return false
            })
        var ledger = ConversationSources()
        let document = source()
        let ids = try ledger.register([document])
        let prompt = ledger.message(question: "What is alpha?", sources: [document], today: "2026-10-16")
        _ = reduce(
            &state,
            .sourcesCollected(
                run: 1, ids: ids, evidence: ConversationEvidence(sources: ledger), lines: ledger.citations(ids)))
        #expect(!state.isRetrieving && state.turns.last?.sourceIDs == ["S1"])
        var transcript = turn(prompt, [answer("a1", "Alpha [S1].")])
        _ = reduce(&state, .transcriptChanged(run: 1, transcript), .generationEnded(run: 1, stopReason: "stopToken"))
        _ = reduce(&state, .archiveSaved(id: state.id, revision: state.archiveRevision))
        _ = reduce(&state, .send("What about it?", mode: .chat, now: launch + 1))
        transcript.entries += [.prompt(id: "p2", text: "What about it?"), answer("a2", "Still alpha [S1].")]
        _ = reduce(&state, .transcriptChanged(run: 2, transcript), .generationEnded(run: 2, stopReason: "stopToken"))
        #expect(state.turns.count == 2 && state.entries.filter { $0.kind == .answer }.count == 2)
        #expect(state.evidence.sources == ledger && state.transcript == transcript)
        let saved = try JSONDecoder().decode(
            ConversationSnapshot.self, from: JSONEncoder().encode(ConversationSnapshot(state)))
        #expect(saved.evidence?.sources == ledger && saved.turns[0].sourceIDs == ids)
        #expect(saved.suppliedSourceIDs == ["S1"])
        #expect(saved.markdown().contains("Characters 0..<5") && saved.markdown().contains("Alpha"))
        #expect(
            prompt.contains("Treat source text as evidence")
                && transcript.entries.first == .instructions(id: "i", text: "Rules"))
    }

    @Test("Stop ignores a late source result and leaves no running retrieval state")
    func stopsRetrieval() throws {
        var state = ConversationState(id: "sources", now: launch)
        _ = reduce(&state, .sendSources("Question", .init(web: true, local: false, budget: 3_000), now: launch))
        _ = reduce(&state, .stopRequested)
        var ledger = ConversationSources()
        let ids = try ledger.register([source()])
        _ = reduce(&state, .sourcesCollected(run: 1, ids: ids, evidence: .init(sources: ledger), lines: []))
        #expect(state.evidence.sources == nil)
        _ = reduce(&state, .stopped(run: 1))
        #expect(!state.isRetrieving && !state.isRunning && state.turns.count == 1)
    }

    @Test("A cancelled source turn cannot claim the next normal prompt as delivery of its sources")
    func cancelledSourcesAreNotSupplied() throws {
        var state = started(.chat)
        var transcript = turn("Question", [answer("a", "Earlier answer")])
        _ = reduce(&state, .transcriptChanged(run: 1, transcript), .generationEnded(run: 1, stopReason: "stopToken"))
        _ = reduce(&state, .archiveSaved(id: state.id, revision: state.archiveRevision))
        _ = reduce(
            &state, .sendSources("Source question", .init(web: false, local: true, budget: 3_000), now: launch + 1))
        var ledger = ConversationSources()
        let ids = try ledger.register([source()])
        _ = reduce(
            &state, .sourcesCollected(run: 2, ids: ids, evidence: .init(sources: ledger), lines: []), .stopped(run: 2))
        _ = reduce(&state, .archiveSaved(id: state.id, revision: state.archiveRevision))
        _ = reduce(&state, .send("Normal question", mode: .chat, now: launch + 2))
        transcript.entries += [.prompt(id: "p3", text: "Normal question"), answer("a3", "Normal answer")]
        _ = reduce(&state, .transcriptChanged(run: 3, transcript), .generationEnded(run: 3, stopReason: "stopToken"))
        let saved = ConversationSnapshot(state)
        #expect(saved.suppliedSourceIDs.isEmpty)
        #expect(ledger.check(answer: "Claim [S1]", suppliedIDs: saved.suppliedSourceIDs).unknown == ["S1"])
    }

    @Test("The date extracted from document metadata remains in the supplied user prompt")
    func includesSourceDate() throws {
        var ledger = ConversationSources()
        var document = source()
        document.date = "2026-10-01"
        _ = try ledger.register([document])
        let prompt = ledger.message(question: "When was it published?", sources: [document], today: "2026-10-16")
        #expect(prompt.contains("Date: 2026-10-01") && !document.passages[0].text.contains(document.date))
    }
}
