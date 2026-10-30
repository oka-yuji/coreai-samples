import ConversationFeature
import Foundation
import Testing

struct ConversationSnapshotTests {
    @Test("Markdown exports preserve fenced blocks inside reasoning and tool results")
    func exportsNestedFences() throws {
        var state = ConversationState(
            id: "nested", now: launch, modelID: "model", instructionsVersion: "i1", toolsVersion: "t1")
        let embedded = "before\n````\n## Embedded heading\n````\nafter"
        let search = tool("nested-tool", "web_search", #"{"query":"example","max_results":1}"#)
        _ = reduce(&state, .send("Read this", mode: .work, now: launch))
        _ = reduce(
            &state,
            .transcriptChanged(
                run: 1,
                TranscriptRecord(entries: [
                    .prompt(id: "p", text: "Read this"), .reasoning(id: "r", text: embedded),
                    calls("c", search), output(search, embedded), answer("a", "Done"),
                ])),
            .generationEnded(run: 1, stopReason: "stopToken"))

        let markdown = ConversationSnapshot(state).markdown()
        let enclosed = "`````\n" + embedded + "\n`````"
        #expect(markdown.components(separatedBy: enclosed).count == 3)
        #expect(markdown.contains("## Assistant\n\nDone"))
    }

    @Test("A snapshot encodes and decodes to an equal value, and its Markdown export lists every turn in order")
    func roundTripsAndExportsEveryTurn() throws {
        var state = ConversationState(
            id: "c1", now: launch, modelID: "qwen3.8-27b", instructionsVersion: "i1", toolsVersion: "t1")
        var entries: [TranscriptRecord.Entry] = [.instructions(id: "i", text: "Rules")]

        _ = reduce(&state, .send("What is Swift?", mode: .chat, now: launch))
        entries += [
            .prompt(id: "p1", text: "What is Swift?"), .reasoning(id: "r1", text: "Easy one."),
            answer("a1", "A programming language."),
        ]
        _ = reduce(
            &state, .transcriptChanged(run: 1, TranscriptRecord(entries: entries)), .stepFinished(run: 1, stats(1)),
            .generationEnded(run: 1, stopReason: "stopToken"))

        let search = tool("t1", "web_search", #"{"max_results":3,"query":"Swift 6.3"}"#)
        let fetch = tool("t2", "fetch_page", #"{"url":"https://swift.org/blog"}"#)
        _ = reduce(&state, .send("Find the release notes", mode: .work, now: launch + 60))
        entries += [
            .prompt(id: "p2", text: "Find the release notes"), calls("c1", search),
            output(search, "1. Swift 6.3 released"), calls("c2", fetch), output(fetch, "Swift 6.3 adds typed throws."),
            answer("a2", "Swift 6.3 adds typed throws."),
        ]
        _ = reduce(
            &state, .transcriptChanged(run: 2, TranscriptRecord(entries: entries)),
            .stepFinished(run: 2, stats(1, calls: 1)), .stepFinished(run: 2, stats(2, calls: 1)),
            .stepFinished(run: 2, stats(3)), .generationEnded(run: 2, stopReason: "stopToken"))

        _ = reduce(&state, .send("Summarize it", mode: .chat, now: launch + 120))
        entries += [.prompt(id: "p3", text: "Summarize it"), answer("a3", "In short, Swift", stop: "maxTokens")]
        let last = reduce(
            &state, .transcriptChanged(run: 3, TranscriptRecord(entries: entries)),
            .stepFinished(run: 3, stats(1, finish: "cap")), .generationEnded(run: 3, stopReason: "maxTokens"))

        let saved = try #require(snapshot(last[2]))
        #expect(saved == ConversationSnapshot(state))
        #expect(
            saved.formatVersion == ConversationSnapshot.currentFormatVersion && saved.status == .completed
                && saved.updatedAt == launch + 120)
        #expect(saved.turns.map(\.mode) == [.chat, .work, .chat] && saved.turns.map(\.start) == [1, 4, 10])
        #expect(saved.metrics.steps == 5 && saved.metrics.promptTokens == 800 && saved.metrics.generatedTokens == 150)
        let data = try JSONEncoder().encode(saved)
        #expect(try JSONDecoder().decode(ConversationSnapshot.self, from: data) == saved)

        let markdown = saved.markdown()
        let outline = markdown.split(separator: "\n").map(String.init).filter { line in
            ["#", "_", ">"].contains { line.hasPrefix($0) }
        }
        let chatStep = "_step 1  |  TTFT 0.50s  |  20.0 tok/s  |  prompt 100  |  reused 50  |  30 tok  |  answered"
        #expect(
            outline == [
                "# Conversation",
                "## You",
                "### Thinking (9 chars)",
                "## Assistant",
                chatStep + "  |  eos_",
                "## You",
                "### Calling web_search",
                "_step 1  |  TTFT 0.50s  |  20.0 tok/s  |  prompt 100  |  reused 50  |  30 tok  |  1 tool call(s)  |  eos_",
                "### web_search result (21 chars)",
                "### Calling fetch_page",
                "_step 2  |  TTFT 0.50s  |  20.0 tok/s  |  prompt 200  |  reused 50  |  30 tok  |  1 tool call(s)  |  eos_",
                "### fetch_page result (28 chars)",
                "## Assistant",
                "_step 3  |  TTFT 0.50s  |  20.0 tok/s  |  prompt 300  |  reused 50  |  30 tok  |  answered  |  eos_",
                "## You",
                "## Assistant",
                chatStep + "  |  cap_",
                "> **Answer cut at the token limit (2048)**: The answer above stops where generation ran out of tokens."
                    + " Raise Max tokens or ask for something shorter.",
            ])
        let order = ["What is Swift?", "Find the release notes", "Summarize it"].compactMap {
            markdown.range(of: "## You\n\n" + $0)?.lowerBound
        }
        #expect(order.count == 3 && order == order.sorted())
        #expect(markdown.contains("````\nquery: Swift 6.3\nmax_results: 3\n````"))
        #expect(markdown.hasPrefix("# Conversation\n\n- title: What is Swift?\n- id: c1\n- model: qwen3.8-27b\n"))
    }

    @Test("Evidence saved before refused fetches were recorded decodes with every attempt unrefused")
    func decodesEvidenceWithoutRefusals() throws {
        let saved = #"{"pages":[],"attempts":[{"url":"https://example.com/gone","ok":false}],"seen":{}}"#
        let evidence = try JSONDecoder().decode(ConversationEvidence.self, from: Data(saved.utf8))
        #expect(evidence.attempts == [ConversationEvidence.Attempt(url: "https://example.com/gone", ok: false)])
        #expect(evidence.attempts.map(\.refused) == [nil])
    }

    @Test("A refused fetch keeps its flag through encoding while other attempts encode without the field")
    func roundTripsRefusedAttempts() throws {
        let evidence = ConversationEvidence(attempts: [
            .init(url: "https://attacker.example/collect?d=1", ok: false, refused: true),
            .init(url: "https://example.com/page", ok: true),
        ])
        let data = try JSONEncoder().encode(evidence)
        #expect(try JSONDecoder().decode(ConversationEvidence.self, from: data) == evidence)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let attempts = try #require(object["attempts"] as? [[String: Any]])
        #expect(attempts.map { $0.keys.sorted() } == [["ok", "refused", "url"], ["ok", "url"]])
    }
}
