import ConversationFeature
import Foundation
import Testing

struct ConversationAccessTests {
    private let now = Date(timeIntervalSince1970: 100)

    @Test(
        "Search off refuses web tools in both modes without changing the conversation",
        arguments: ConversationMode.allCases)
    func offDeniesWeb(_ mode: ConversationMode) {
        var state = ConversationState(id: "access", now: now)
        let history = TranscriptRecord(entries: [.prompt(id: "old", text: "Earlier question")])
        state.transcript = history
        _ = ConversationFeature.reduce(&state, .send("Answer from memory", mode: mode, now: now))
        for name in ["web_search", "fetch_page"] {
            let call = ToolCallRecord(id: name, name: name, arguments: "{}")
            let effects = ConversationFeature.reduce(&state, .toolRequested(run: state.runID, call))
            #expect(!effects.contains { if case .run(.runTool, _) = $0 { true } else { false } })
            #expect(effects.contains { if case .run(.refuseTool, _) = $0 { true } else { false } })
        }
        #expect(state.transcript == history && state.mode == mode)
        #expect(!state.refusedSave)
    }

    @Test("Work source requests retain Work mode and note tools", arguments: [false, true])
    func workSources(_ web: Bool) throws {
        var state = ConversationState(id: "access", now: now, mode: .work)
        let request = ConversationSourceRequest(web: web, local: true, budget: 3_000)
        _ = ConversationFeature.reduce(&state, .sendSources("Research and save a note", request, now: now))
        let policy = try #require(state.turns.last?.policy)
        #expect(state.mode == .work && policy.mode == .work && policy.sourceRequest == request)
        #expect(policy.allowedTools.contains("write_note"))
        #expect(policy.allowedTools.contains("web_search") == web)
        #expect(policy.allowedTools.contains("fetch_page") == web)
    }

    @Test("Chat source answers prohibit tools while Work prompts permit note saving")
    func sourceModes() throws {
        let request = ConversationSourceRequest(web: true, local: false, budget: 6_000)
        var state = ConversationState(id: "access", now: now)
        _ = ConversationFeature.reduce(&state, .sendSources("Explain", request, now: now))
        #expect(state.mode == .chat && state.turns.last?.policy.allowedTools.isEmpty == true)
        let source = ConversationSource(
            document: "file:///manual.md", version: String(repeating: "a", count: 64), title: "Manual",
            passages: [SourcePassage(offset: 0, text: "Alpha")])
        let ledger = ConversationSources()
        let prepared = try #require(
            try ledger.prepare(
                question: "Save a summary", sources: [source], today: "2026-10-18", budget: 4_000,
                count: { $0.count }, mode: .work))
        #expect(prepared.prompt.contains("permitted note tools") && !prepared.prompt.contains("Do not call tools"))
        var checked = ledger
        _ = try checked.register([source])
        #expect(checked.check(answer: "Uncited answer", notes: ["Note [S1]"], required: true).missing)
    }

    @Test(
        "Continue and repair intersect the original web permission with the current setting", arguments: [false, true])
    func continuationAccess(_ originallyAllowed: Bool) throws {
        var state = ConversationState(
            id: "access", now: now,
            settings: ConversationSettings(webSearchEnabled: originallyAllowed))
        _ = ConversationFeature.reduce(&state, .send("Write a report", mode: .work, now: now))
        _ = ConversationFeature.reduce(
            &state,
            .transcriptChanged(
                run: 1,
                TranscriptRecord(entries: [
                    .prompt(id: "p", text: "Write a report"),
                    .response(id: "a", text: "Partial report", metadata: StepMetadata(stopReason: "maxTokens")),
                ])))
        _ = ConversationFeature.reduce(&state, .generationEnded(run: 1, stopReason: "maxTokens"))
        _ = ConversationFeature.reduce(&state, .archiveSaved(id: state.id, revision: state.archiveRevision))
        _ = ConversationFeature.reduce(
            &state, .settingsChanged(ConversationSettings(webSearchEnabled: !originallyAllowed)))
        var repair = state
        _ = ConversationFeature.reduce(&state, .continueRequested(now: now + 1))
        #expect(state.turns.count == 2 && state.turns.last?.settings.webSearchEnabled == false)
        #expect(state.turns.last?.policy.allowedTools.contains("write_note") == true)
        _ = ConversationFeature.reduce(&repair, .sendRepair("Repair", now: now + 1))
        #expect(repair.turns.count == 2 && repair.turns.last?.settings.webSearchEnabled == false)
    }

    @Test("Missing legacy permissions decode as search off and new archives retain the explicit choice")
    func archivePermission() throws {
        var state = ConversationState(id: "access", now: now, settings: ConversationSettings(webSearchEnabled: true))
        _ = ConversationFeature.reduce(&state, .send("Question", mode: .work, now: now))
        let encoded = try JSONEncoder().encode(ConversationSnapshot(state))
        let roundTrip = try JSONDecoder().decode(ConversationSnapshot.self, from: encoded)
        #expect(roundTrip.turns[0].settings.webSearchEnabled == true)
        var object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        var turns = try #require(object["turns"] as? [[String: Any]])
        var settings = try #require(turns[0]["settings"] as? [String: Any])
        settings.removeValue(forKey: "webSearchEnabled")
        turns[0]["settings"] = settings
        object["turns"] = turns
        object["formatVersion"] = 4
        let legacy = try JSONDecoder().decode(
            ConversationSnapshot.self, from: JSONSerialization.data(withJSONObject: object))
        #expect(!legacy.turns[0].policy.allowedTools.contains("web_search"))
        #expect(legacy.turns[0].policy.allowedTools.contains("write_note"))
    }

    @Test("Invalid source selections and late retrieval results cannot start or change a turn")
    func invalidSources() {
        var state = ConversationState(id: "access", now: now, mode: .work)
        let before = state
        #expect(
            ConversationFeature.reduce(
                &state, .sendSources("Question", .init(web: false, local: false, budget: 3_000), now: now)
            ).isEmpty)
        #expect(state == before)
        _ = ConversationFeature.reduce(
            &state, .sendSources("Question", .init(web: false, local: true, budget: 3_000), now: now))
        _ = ConversationFeature.reduce(&state, .stopRequested)
        let stopping = state
        #expect(
            ConversationFeature.reduce(
                &state, .sourcesCollected(run: state.runID, ids: ["S1"], evidence: ConversationEvidence(), lines: [])
            ).isEmpty)
        #expect(state == stopping)
    }

    @Test("A Work source cutoff reuses collected sources without requesting retrieval again")
    func sourceBudget() throws {
        var state = ConversationState(id: "access", now: now, mode: .work, settings: ConversationSettings(maxSteps: 1))
        _ = ConversationFeature.reduce(
            &state, .sendSources("Save a summary", .init(web: false, local: true, budget: 3_000), now: now))
        _ = ConversationFeature.reduce(
            &state, .sourcesCollected(run: 1, ids: ["S1"], evidence: ConversationEvidence(), lines: []))
        let call = ToolCallRecord(id: "save", name: "write_note", arguments: "{}")
        _ = ConversationFeature.reduce(&state, .toolRequested(run: 1, call))
        let effects = ConversationFeature.reduce(&state, .budgetStopped(run: 1))
        let policy = try #require(state.turns.last?.policy)
        #expect(policy.answerOnly && policy.sourceRequest == nil && policy.sourceIDs == ["S1"])
        #expect(policy.allowedTools.isEmpty && !state.isRetrieving)
        #expect(
            effects.contains {
                if case .run(.generate(_, _, let final, _), _) = $0 {
                    return final.sourceRequest == nil && final.answerOnly
                }
                return false
            })
    }
}
