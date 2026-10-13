import CryptoKit
import Foundation
import FoundationModels
import RAGContext

enum AgentSelfTest {
    static func run() -> Never {
        let offline = LaunchArguments.flag("--offline")
        Task { @MainActor in exit(await execute(offline: offline)) }
        RunLoop.main.run()
        fatalError("the main run loop returned")
    }

    static func runEndToEnd() -> Never {
        let effort: AgentReasoningEffort
        if let requested = LaunchArguments.value("--effort") {
            guard let parsed = AgentReasoningEffort(rawValue: requested) else {
                out("agent-e2e: unknown effort '\(requested)' (off|low|medium|xhigh)")
                exit(2)
            }
            effort = parsed
        } else {
            effort = .low
        }
        Task { @MainActor in
            let code = await executeEndToEnd(
                effort: effort, deadline: LaunchArguments.value("--deadline").flatMap { Double($0) } ?? 900)
            await MetricsLog.flush()
            exit(code)
        }
        RunLoop.main.run()
        fatalError("the main run loop returned")
    }

    @MainActor
    private static func executeEndToEnd(effort: AgentReasoningEffort, deadline: Double) async -> Int32 {
        guard let model = LaunchArguments.value("--model"), let task = LaunchArguments.value("--task") else {
            out("agent-e2e: --model <bundle> and --task \"…\" are required")
            return 2
        }
        let agent = AgentViewModel()
        agent.persistsSettings = false
        let m4g: String?
        do {
            m4g = try applyM4GOptions(to: agent)
        } catch {
            out("agent-e2e: " + ((error as? AgentToolMessage)?.text ?? "\(error)"))
            return 2
        }
        let chat = ChatViewModel()
        out("[e2e] loading \(model)")
        let loadStart = Date()
        await chat.load(URL(filePath: model, directoryHint: .isDirectory))
        guard chat.phase == .ready, let stats = chat.loadStats else {
            out("agent-e2e: load failed — \(chat.phase)")
            return 1
        }
        out(String(format: "[e2e] loaded %@ in %.1fs", chat.modelName, Date().timeIntervalSince(loadStart)))
        out(
            String(
                format: "[e2e] prepared: cached=%@ specializeSeconds=%.2f contextLimit=%d", String(stats.cached),
                stats.specializeSeconds,
                chat.contextLimit))
        agent.effort = effort
        if let steps = LaunchArguments.value("--max-steps").flatMap({ Int($0) }) { agent.maxIterations = steps }
        if let tokens = LaunchArguments.value("--max-tokens").flatMap({ Int($0) }) { agent.maxNewTokens = tokens }
        if let budget = LaunchArguments.value("--page-budget").flatMap({ Int($0) }) {
            agent.pageCharacterBudget = budget
        }
        agent.allowFileOperations = LaunchArguments.flag("--file-ops")
        if let notes = LaunchArguments.value("--notes-dir") { agent.notesFolderPath = notes }
        out(
            "[e2e] effort=\(agent.effort.rawValue) steps=\(agent.maxIterations) tokens=\(agent.maxNewTokens) "
                + "pageBudget=\(agent.pageCharacterBudget) search=\(agent.searchProvider) date=\(agent.fixedDate ?? AgentPromptBuilder.today()) "
                + "fileOps=\(agent.allowFileOperations) notesDir=\(agent.notesDirectory.path(percentEncoded: false))")
        if let m4g { out("[e2e] m4g: \(m4g)") }
        out("[e2e] task: \(task)")
        agent.input = task
        let started = Date()
        agent.deadline = started.addingTimeInterval(deadline)
        guard agent.send(chat: chat) else {
            out("agent-e2e: the agent refused to start")
            return 1
        }
        var printed: Set<String> = []
        var status = ""
        var stoppedByDeadline = false
        func printSettledEntries() {
            for entry in agent.entries where !entry.live && !printed.contains(entry.id) {
                printed.insert(entry.id)
                out("--- \(entry.kind) \(entry.title)\n\(entry.text)")
                if let stats = entry.stats {
                    status = stats
                    out("[e2e] \(stats)")
                }
            }
        }
        while agent.running {
            try? await Task.sleep(for: .milliseconds(500))
            printSettledEntries()
            if agent.statusLine != status {
                status = agent.statusLine
                out("[e2e] \(status)")
            }
            if Date().timeIntervalSince(started) > deadline {
                out("[e2e] deadline of \(Int(deadline))s reached — stopping")
                stoppedByDeadline = true
                agent.stop()
                break
            }
        }
        while agent.running { try? await Task.sleep(for: .milliseconds(500)) }
        printSettledEntries()
        let failureTitles = agent.entries.filter { $0.kind == .failure }.map(\.title)
        out(
            String(
                format: "[e2e] %.1fs  toolCalls=%d  entries=%d  failures=%d", Date().timeIntervalSince(started),
                agent.entries.filter { $0.kind == .toolCall }.count, agent.entries.count, failureTitles.count))
        out("[e2e] status: \(agent.statusLine)")
        let answer = agent.finalAnswer()
        if let before = agent.answerBeforeRepair {
            out("[e2e] answer before repair (\(before.count) chars):\n\(before)")
        }
        out("[e2e] final answer (\(answer.count) chars):\n\(answer)")
        let outcome = e2eOutcome(
            stoppedByDeadline: stoppedByDeadline, failureTitles: failureTitles, answer: answer,
            finishReason: agent.lastFinishReason,
            held: agent.guardOutcome == heldOutcome)
        out("[e2e] outcome: \(outcome)")
        for check in agent.checks { out(check.logLine) }
        if let guardOutcome = agent.guardOutcome {
            out(
                String(
                    format:
                        "[guard] outcome=%@ reason=%@ repair_turns=%d repair_fetches=%d repair_seconds=%.1f repair_prompt_tokens=%d "
                        + "repair_generated_tokens=%d drafts=%d",
                    guardOutcome, agent.guardReason, agent.repairTurns, agent.repairFetches, agent.repairSeconds ?? 0,
                    agent.repairPromptTokens ?? 0, agent.repairGeneratedTokens ?? 0, agent.draftOrder.count))
        }
        if let cassette = agent.cassette {
            out(
                "[cassette] summary hits=\(await cassette.hits) misses=\(await cassette.misses) records=\(await cassette.records)"
            )
        }
        let completed = outcome == answeredOutcome || outcome == heldOutcome
        out(
            (completed ? "AGENT E2E COMPLETED" : "AGENT E2E INCOMPLETE")
                + "  diverges=\(agent.divergeCount)  capSteps=\(agent.capSteps)")
        return completed ? 0 : 1
    }

    @MainActor
    private static func applyM4GOptions(
        to agent: AgentViewModel, arguments: [String] = CommandLine.arguments
    ) throws -> String? {
        func choice(_ name: String, _ allowed: [String]) throws -> String? {
            guard let value = LaunchArguments.value(name, in: arguments) else { return nil }
            guard allowed.contains(value) else {
                throw AgentToolMessage(text: "\(name) must be \(allowed.joined(separator: "|")), not '\(value)'")
            }
            return value
        }
        var given: [String] = []
        if let value = try choice("--instructions", ["frozen", "cite-fetched"]) {
            agent.citeFetched = value == "cite-fetched"
            given.append("instructions=\(value)")
        }
        if let value = try choice("--citation-guard", ["on", "off"]) {
            agent.citationGuard = value == "on"
            given.append("citationGuard=\(value)")
        }
        if let value = try choice("--sources-required", ["yes", "no", "auto"]) {
            agent.sourcesRequired = value == "auto" ? nil : value == "yes"
            given.append("sourcesRequired=\(value)")
        }
        if let value = LaunchArguments.value("--date", in: arguments) {
            guard AgentPromptBuilder.isDateLabel(value) else {
                throw AgentToolMessage(
                    text:
                        "--date must read like \"\(AgentPromptBuilder.today())\" with the right weekday, not '\(value)'"
                )
            }
            agent.fixedDate = value
            given.append("date=\(value)")
        }
        if LaunchArguments.flag("--stable-note-paths", in: arguments) {
            agent.stableNotePaths = true
            given.append("stableNotePaths=true")
        }
        let latency = try choice("--cassette-latency", ["replay", "none"])
        let mode = try choice("--cassette-mode", ToolCassette.Mode.allCases.map(\.rawValue)).flatMap(
            ToolCassette.Mode.init(rawValue:))
        let path = LaunchArguments.value("--tool-cassette", in: arguments)
        if path != nil || mode != nil {
            guard let path, let mode else {
                throw AgentToolMessage(
                    text: "--tool-cassette <path> and --cassette-mode record|replay|replay-or-record go together")
            }
            let cassette = try ToolCassette(file: URL(filePath: path), mode: mode, replaysLatency: latency != "none")
            agent.cassette = cassette
            given.append(
                "cassette=\(path) mode=\(mode.rawValue) latency=\(latency ?? "replay") entries=\(cassette.loaded)")
        } else if latency != nil {
            throw AgentToolMessage(text: "--cassette-latency needs --tool-cassette")
        }
        return given.isEmpty ? nil : given.joined(separator: " ")
    }

    static let answeredOutcome = "answered"
    static let heldOutcome = "held"

    static func e2eOutcome(
        stoppedByDeadline: Bool, failureTitles: [String], answer: String, finishReason: String?, held: Bool = false
    ) -> String {
        if held { return heldOutcome }
        if stoppedByDeadline { return "stopped" }
        if let title = failureTitles.first {
            if title.hasPrefix("Context window is full") { return "overflow" }
            if title.hasPrefix("Step limit reached") { return "step-limit" }
            if title.hasPrefix("Answer cut at the token limit") { return "cap" }
            if title.hasPrefix("Stopped") { return "stopped" }
            return "failed"
        }
        if answer.isEmpty || answer == AgentViewModel.noAnswerText { return "failed" }
        return finishReason == "eos" ? answeredOutcome : "failed"
    }

    private static func out(_ text: String) { FileHandle.standardOutput.write(Data((text + "\n").utf8)) }

    @MainActor
    static func execute(offline: Bool) async -> Int32 {
        var failures = 0
        func check(_ name: String, _ passed: Bool, _ detail: @autoclosure () -> String = "") {
            if passed {
                out("PASS  \(name)")
            } else {
                failures += 1
                out("FAIL  \(name)  \(detail())")
            }
        }

        let golden =
            DateComponents(calendar: Calendar(identifier: .gregorian), year: 2026, month: 8, day: 26, hour: 12).date
            ?? .now
        check("prompt/today", AgentPromptBuilder.today(golden) == goldenDate, AgentPromptBuilder.today(golden))
        let instructions = AgentPromptBuilder.instructions(date: goldenDate)
        check(
            "prompt/instructions",
            instructions.hasPrefix(
                "You are an agent running on this device. Use the tools to gather facts before you answer.")
                && instructions.contains("answer directly without calling a tool.\n\n- If a tool returns an error")
                && instructions.components(separatedBy: "\n- ").count == 4
                && instructions.contains("end the answer with a \"Sources:\" section")
                && instructions.hasSuffix(
                    "past tense and prefer the most recent sources.\n\nCurrent date: " + goldenDate),
            instructions.debugDescription)

        let run: @Sendable (AgentToolCall) async -> String = { call in
            await AgentTools.run(
                call, settings: AgentToolSettings(notesDirectory: FileManager.default.temporaryDirectory)
            ).text
        }
        let plain = AgentPromptBuilder.tools(fileOperations: false, run: run)
        let withFiles = AgentPromptBuilder.tools(fileOperations: true, run: run)
        let search = plain.first.flatMap {
            try? JSONSerialization.jsonObject(with: JSONEncoder().encode($0.parameters)) as? [String: Any]
        }
        let properties = search?["properties"] as? [String: [String: Any]]
        check(
            "tools/definitions",
            plain.map(\.name) == ["web_search", "fetch_page", "write_note"]
                && withFiles.map(\.name) == ["web_search", "fetch_page", "write_note", "move_note"]
                && plain.first?.description == "Search the web and return the top results as title, url and snippet."
                && search?["required"] as? [String] == ["query"]
                && search?["x-order"] as? [String] == ["query", "max_results"]
                && properties?["query"]?["description"] as? String == "What to search for."
                && properties?["max_results"]?["type"] as? String == "integer",
            "\(plain.map(\.name)) \(String(describing: search))")

        let content = try? GeneratedContent(json: #"{"max_results": 5, "query": "MLX"}"#)
        let arguments = content.map { AgentPromptBuilder.arguments("web_search", $0) } ?? []
        check(
            "tools/argumentsInDefinitionOrder",
            arguments == [AgentArgument(name: "query", value: "MLX"), AgentArgument(name: "max_results", value: "5")],
            "\(arguments)")

        let fetch = AgentTool<FetchPageArguments>(name: "fetch_page", description: "", run: run)
        var refused = "no arguments"
        if let bad = try? FetchPageArguments(GeneratedContent(json: #"{"url": "not a url"}"#)) {
            refused = await fetch.call(arguments: bad)
        }
        check("tools/failuresAreText", refused == "error: fetch_page needs an absolute http or https URL", refused)

        let budget = 1_200
        let head = AgentTools.clip(longPageFixture, budget: budget)
        let selected = AgentTools.pageText(
            longPageFixture,
            query: "\u{30D0}\u{30C3}\u{30C1}\u{63A8}\u{8AD6}\u{306E}\u{30BF}\u{30A4}\u{30E0}\u{30A2}\u{30A6}\u{30C8}",
            budget: budget)
        check(
            "fetch/queryFindsLatePassage",
            !head.contains("45 \u{79D2}") && selected.contains("45 \u{79D2}") && selected.contains("\u{4ED8}\u{9332}B")
                && selected.hasPrefix("Core ML \u{30C9}\u{30AD}\u{30E5}\u{30E1}\u{30F3}\u{30C8} \u{76EE}\u{6B21}")
                && selected.contains("\n…\n")
                && selected.count <= budget + 200,
            "head=\(head.contains("45 \u{79D2}")) selected=\(selected.count) chars: "
                + selected.prefix(120).debugDescription)

        check(
            "fetch/noQueryIsUnchanged",
            AgentTools.pageText(longPageFixture, query: "", budget: budget) == head
                && AgentTools.pageText(longPageFixture, query: "   ", budget: budget) == head,
            "a page fetched without a query must still be the plain head")

        let hostileQuery =
            "\"\u{30D0}\u{30C3}\u{30C1}\u{63A8}\u{8AD6}\" -\u{306E} AND* (\u{30BF}\u{30A4}\u{30E0}\u{30A2}\u{30A6}\u{30C8}) col:on"
        let hostileResult = AgentTools.pageText(longPageFixture, query: hostileQuery, budget: budget)
        check(
            "fetch/hostileQuerySurvives",
            hostileResult.contains("45 \u{79D2}")
                && AgentTools.matchTerms(hostileQuery).allSatisfy { !$0.contains("\"") }
                && AgentTools.pageText(longPageFixture, query: "?!()", budget: budget) == head,
            "terms=\(AgentTools.matchTerms(hostileQuery))")

        let overlapping = AgentViewModel.gramOverlap("abcdefghij", "zz abcdefghi zz")
        let disjoint = AgentViewModel.gramOverlap("abcdefghij", "0123456789")
        let answer = "Core ML runs on device.\n\nSources:\nhttps://example.com/read\n- https://example.org/never-opened"
        let split = AgentViewModel.splitSources(answer)
        check(
            "cite/overlapAndUnseenURL",
            abs(overlapping - 2.0 / 3.0) < 1e-9 && disjoint == 0
                && split.urls == ["https://example.com/read", "https://example.org/never-opened"]
                && split.body == "Core ML runs on device.\n"
                && AgentViewModel.normalized("https://Example.com/read/") == "https://example.com/read",
            "overlap=\(overlapping) disjoint=\(disjoint) urls=\(split.urls) body=\(split.body.debugDescription)")

        let stripped = AgentTools.plainText(fromHTML: htmlFixture)
        check("html/strip", stripped == "Hello & welcome\nIt's here\nTail — end", stripped.debugDescription)

        let entities = AgentTools.decodeEntities("a&amp;b &#65;&#x42; &nbsp;&unknown;")
        check("html/entities", entities == "a&b AB  &unknown;", entities.debugDescription)

        let results = AgentTools.parseDuckDuckGo(duckDuckGoFixture)
        check(
            "search/duckDuckGoFixture",
            results == [
                AgentTools.SearchResult(
                    title: "Core ML | Apple Developer Documentation",
                    url: "https://developer.apple.com/documentation/coreml",
                    snippet: "Use Core ML on a person's device."),
                AgentTools.SearchResult(
                    title: "Core ML Tools", url: "https://example.com/core-ml", snippet: "Convert models & run them."),
                AgentTools.SearchResult(title: "No snippet here", url: "https://example.org/bare", snippet: ""),
                AgentTools.SearchResult(
                    title: "Fourth result", url: "https://example.net/fourth", snippet: "Fourth snippet."),
            ], "\(results)")

        check(
            "search/rateLimited",
            AgentTools.searchStatusError(202, brave: false)
                == "error: search backend returned HTTP 202 (rate limited or blocked). Do not retry the "
                + "same query immediately; try a different tool or rephrase once."
                && AgentTools.searchStatusError(429, brave: false)?.contains("HTTP 429") == true
                && AgentTools.searchStatusError(401, brave: true)?.contains("Brave API key rejected") == true
                && AgentTools.searchStatusError(402, brave: true)?.contains("paid plan") == true,
            AgentTools.searchStatusError(202, brave: false) ?? "nil")

        check(
            "search/ok200",
            AgentTools.searchStatusError(200, brave: false) == nil
                && AgentTools.searchStatusError(200, brave: true) == nil, "200 must not be an error")

        let hostile = "../../etc/pas swd: <hack>"
        let hostileName = AgentTools.noteFileName(hostile)
        check(
            "notes/fileName",
            !hostileName.contains("/") && !hostileName.contains("..")
                && hostileName.hasSuffix(".md") && AgentTools.noteFileName("   ") == "note.md"
                && AgentTools.noteFileName(String(repeating: "a", count: 300)).count == 83, hostileName)
        do {
            let path = try AgentTools.writeNote(title: hostile, content: "body")
            let inside =
                URL(fileURLWithPath: path).deletingLastPathComponent().standardizedFileURL
                == AgentTools.defaultNotesDirectory().standardizedFileURL
            check(
                "notes/write", inside && FileManager.default.fileExists(atPath: path) && path.contains("/agent-notes/"),
                path)
            let second = try AgentTools.writeNote(title: hostile, content: "other")
            let third = try AgentTools.writeNote(title: hostile, content: "more")
            let kept = (try? String(contentsOf: URL(fileURLWithPath: path), encoding: .utf8)) ?? ""
            let base = String(path.dropLast(3))
            check(
                "notes/writeSuffixesOnCollision",
                second == base + " 2.md" && third == base + " 3.md" && kept.contains("body") && !kept.contains("other"),
                second)
            try? FileManager.default.removeItem(atPath: third)
            try? FileManager.default.removeItem(atPath: second)
            try? FileManager.default.removeItem(atPath: path)
        } catch {
            check("notes/write", false, "\(error)")
            check("notes/writeSuffixesOnCollision", false, "\(error)")
        }

        let home = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true).standardizedFileURL
            .resolvingSymlinksInPath()
        let rootsFromHome = AgentTools.defaultFileRoots(notesDirectory: home)
        let rootsFromRoot = AgentTools.defaultFileRoots(notesDirectory: URL(fileURLWithPath: "/", isDirectory: true))
        check(
            "notes/rootsExcludeHome",
            !rootsFromHome.contains(home) && rootsFromHome.count == 3 && rootsFromRoot.count == 3
                && AgentTools.defaultFileRoots(notesDirectory: AgentTools.defaultNotesDirectory()).count == 4,
            "\(rootsFromHome.map { $0.lastPathComponent })")

        let movesIn = FileManager.default.temporaryDirectory
            .appending(path: "agent-selftest-notes-\(UUID().uuidString)", directoryHint: .isDirectory)
        let archive = movesIn.appending(path: "archive", directoryHint: .isDirectory)
        func refusal(_ body: () throws -> String) -> String {
            do { return "moved to " + (try body()) } catch let failure as AgentToolMessage {
                return failure.text
            } catch {
                return "\(error)"
            }
        }
        do {
            let saved = try AgentTools.writeNote(title: "Move me", content: "body", directory: movesIn)
            let moved = try AgentTools.moveNote(
                title: "Move me", destination: archive.path(percentEncoded: false), notesDirectory: movesIn)
            check(
                "notes/move",
                moved.hasSuffix("/archive/Move me.md") && FileManager.default.fileExists(atPath: moved)
                    && !FileManager.default.fileExists(atPath: saved), moved)
        } catch {
            check("notes/move", false, "\(error)")
        }
        _ = try? AgentTools.writeNote(title: "Move me", content: "body", directory: movesIn)
        var refusals = [
            refusal {
                try AgentTools.moveNote(
                    title: "Move me", destination: "/tmp/agent-refuse", notesDirectory: movesIn,
                    allowedRoots: [movesIn.standardizedFileURL.resolvingSymlinksInPath()])
            },
            refusal {
                try AgentTools.moveNote(
                    title: "Move me", destination: archive.path(percentEncoded: false), notesDirectory: movesIn)
            },
        ]
        refusals.append(
            await AgentTools.run(
                AgentToolCall(
                    name: "move_note",
                    arguments: [
                        AgentArgument(name: "title", value: "Move me"),
                        AgentArgument(name: "destination", value: "Desktop"),
                    ]),
                settings: AgentToolSettings(notesDirectory: movesIn)
            ).text)
        check("notes/moveRefused", refusals.allSatisfy { $0.hasPrefix("error:") }, refusals.joined(separator: " | "))
        try? FileManager.default.removeItem(at: movesIn)

        let capTitle = "Answer cut at the token limit (1536)"
        check(
            "e2e/outcomeVerdict",
            e2eOutcome(stoppedByDeadline: false, failureTitles: [], answer: "done", finishReason: "eos") == "answered"
                && e2eOutcome(stoppedByDeadline: true, failureTitles: [], answer: "done", finishReason: "eos")
                    == "stopped"
                && e2eOutcome(
                    stoppedByDeadline: false, failureTitles: [], answer: AgentViewModel.noAnswerText,
                    finishReason: "eos") == "failed"
                && e2eOutcome(stoppedByDeadline: false, failureTitles: [], answer: "done", finishReason: "cap")
                    == "failed"
                && e2eOutcome(stoppedByDeadline: false, failureTitles: [], answer: "done", finishReason: nil)
                    == "failed"
                && e2eOutcome(
                    stoppedByDeadline: false, failureTitles: ["Step limit reached"], answer: "done", finishReason: "eos"
                ) == "step-limit"
                && e2eOutcome(
                    stoppedByDeadline: false, failureTitles: ["Context window is full"], answer: "done",
                    finishReason: "eos") == "overflow"
                && e2eOutcome(stoppedByDeadline: false, failureTitles: [capTitle], answer: "done", finishReason: "cap")
                    == "cap"
                && e2eOutcome(stoppedByDeadline: false, failureTitles: ["Stopped"], answer: "", finishReason: nil)
                    == "stopped"
                && e2eOutcome(
                    stoppedByDeadline: false, failureTitles: ["Generation failed"], answer: "done", finishReason: "eos")
                    == "failed",
            "verdict table changed")

        let userText = "who ships Core ML bundles?"
        let thinkText = "I should search before answering."
        let callText = "query: Core ML bundles"
        let resultText = "1. Result\n```\nfenced block inside the tool result\n```\ntail line"
        let answerText = "Apple ships them."
        let statsText = "step 1  |  TTFT 0.50s  |  12.3 tok/s"
        let noteText = "1/1 cited URLs were retrieved in this task"
        let agent = AgentViewModel()
        agent.persistsSettings = false
        agent.entries = [
            .init(kind: .user, title: "You", text: userText),
            .init(kind: .thinking, title: "Thinking", text: thinkText),
            .init(kind: .toolCall, title: "web_search", text: callText),
            .init(kind: .toolResult, title: "web_search result", text: resultText),
            .init(kind: .answer, title: "Assistant", text: answerText, stats: statsText),
            .init(kind: .note, title: "Citations", text: noteText),
        ]
        let markdown = agent.transcriptMarkdown()
        let offsets = [userText, thinkText, callText, resultText, answerText, noteText].map {
            markdown.range(of: $0).map { markdown.distance(from: markdown.startIndex, to: $0.lowerBound) }
        }
        let present = offsets.compactMap { $0 }
        check(
            "transcript/markdown",
            present.count == offsets.count && present == present.sorted()
                && markdown.contains("### Thinking (\(thinkText.count) chars)")
                && markdown.contains("### web_search result (\(resultText.count) chars)")
                && markdown.contains("### Calling web_search")
                && markdown.contains("\n_\(statsText)_\n")
                && markdown.contains("````"),
            "offsets \(offsets)")

        await m4gChecks { name, passed, detail in check(name, passed, detail) }
        await fetchAddressChecks(offline: offline) { name, passed, detail in check(name, passed, detail) }
        if !offline, let directory = LaunchArguments.value("--fetch-capture-dir") {
            await captureFetchPages(directory) { name, passed, detail in check(name, passed, detail) }
        }
        for result in await HubDownloadSelfTest.run(offline: offline) {
            check(result.name, result.passed, result.detail)
        }
        await ModelSelectionSelfTest.run { name, passed, detail in check(name, passed, detail) }

        if offline {
            out("SKIP  search/live (--offline)")
            out("SKIP  html/webView (--offline)")
        } else {
            do {
                let live = try await AgentTools.duckDuckGoResults(query: "Core ML", limit: 5)
                if live.isEmpty {
                    out("SKIP  search/live (no results — network or endpoint change)")
                } else {
                    check(
                        "search/live", live.allSatisfy { !$0.title.isEmpty && $0.url.hasPrefix("http") },
                        "\(live.prefix(2))")
                    out("      first: \(live[0].title) \(live[0].url)")
                }
            } catch {
                out("SKIP  search/live (\(error))")
            }
            for address in ["https://developer.apple.com/documentation/coreml", "https://tenki.jp/forecast/3/16/"] {
                guard let url = URL(string: address) else { continue }
                do {
                    let live = AgentTools.condense(try await AgentPageReader.readableText(url: url))
                    check("html/webView \(url.host() ?? address)", live.count > 400, "only \(live.count) chars")
                    out(
                        "      webView: \(live.count) chars, first line: "
                            + (live.split(separator: "\n").first.map(String.init) ?? ""))
                } catch {
                    out("SKIP  html/webView \(url.host() ?? address) (\(error))")
                }
            }
        }

        out(failures == 0 ? "AGENT SELFTEST PASS" : "AGENT SELFTEST FAIL (\(failures))")
        return failures == 0 ? 0 : 1
    }

    static let goldenDate = "2026-08-26 (Wed)"

    static let frozenInstructionsSHA256 = "0fb923293ac863625dcae287a826fb5cdaaedd57ab39eefff2f7b979ec3cc430"

    static let appendixSHA256 = [
        "7a22fe768ac577131d7df8e6fe0592a89149852837e2e84415a16c5ad8788b1d",
        "60bcc151dd2e931e53f1e9859ffdc7ad144a6c4e680b280a9169b1c51d992b49",
        "3a6fa54009bb644d3bf299341a00831fd369a40eb029787b6c449fd1dc6f1b94",
    ]

    static func sha256(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    @MainActor
    private static func m4gChecks(_ check: (String, Bool, String) -> Void) async {
        let frozen = AgentPromptBuilder.instructions(date: goldenDate)
        check(
            "m4g/frozenInstructionsUnchanged", sha256(frozen) == frozenInstructionsSHA256 && frozen.utf8.count == 986,
            sha256(frozen))
        let cited = AgentPromptBuilder.instructions(date: goldenDate, citeFetched: true)
        let added = AgentPromptBuilder.citeFetchedInstructions
        check(
            "m4g/citeFetchedInstructions",
            cited
                == frozen.replacingOccurrences(
                    of: "\n\nCurrent date: ", with: "\n" + added.joined(separator: "\n") + "\n\nCurrent date: ")
                && (added + [AgentPromptBuilder.repairPromptTemplate]).map(sha256) == appendixSHA256,
            cited.debugDescription)
        let twoURLs = AgentPromptBuilder.repairPrompt(unfetched: ["https://a.example/x", "https://b.example/y"])
        let noURL = AgentPromptBuilder.repairPrompt(unfetched: [])
        check(
            "m4g/repairPrompt",
            twoURLs.hasPrefix(
                "Citation check by the app: these sources were not opened with fetch_page in this task: "
                    + "https://a.example/x, https://b.example/y. Open the pages you need")
                && noURL.contains("in this task: none listed. Open") && !noURL.contains("<URL list"),
            twoURLs)
        check(
            "m4g/dateLabel",
            AgentPromptBuilder.isDateLabel("2026-10-04 (Sun)") && !AgentPromptBuilder.isDateLabel("2026-10-04 (Mon)")
                && !AgentPromptBuilder.isDateLabel("2026-10-04")
                && AgentPromptBuilder.isDateLabel(AgentPromptBuilder.today()),
            "")

        let readings: [(String, [String])] = [
            ("see https://ja.wikipedia.org/wiki/Metal_(API).", ["https://ja.wikipedia.org/wiki/Metal_(API)"]),
            (
                "(see https://en.wikipedia.org/wiki/MLX_(machine_learning_framework))",
                ["https://en.wikipedia.org/wiki/MLX_(machine_learning_framework)"]
            ),
            (
                "(https://example.com/a) and [t](https://example.com/b)",
                ["https://example.com/a", "https://example.com/b"]
            ),
            ("[https://example.com/c](https://example.com/c/)", ["https://example.com/c"]),
            (
                "\u{51FA}\u{5178}\u{FF1A}https://example.jp/x\u{3002}\u{FF08}https://example.jp/y\u{FF09}\u{300C}https://example.jp/z\u{300D}",
                ["https://example.jp/x", "https://example.jp/y", "https://example.jp/z"]
            ),
            (
                "<https://example.com/d>, **https://example.com/e**; https://example.com/f?",
                ["https://example.com/d", "https://example.com/e", "https://example.com/f"]
            ),
            ("HTTPS://Example.com/Path?Q=1#frag", ["HTTPS://Example.com/Path?Q=1#frag"]),
            ("no link, just https:// and text", []),
        ]
        let misread = readings.filter { AgentViewModel.guardURLs(in: $0.0) != $0.1 }
        check(
            "m4g/guardURLReader",
            misread.isEmpty
                && AgentViewModel.urls(in: "https://ja.wikipedia.org/wiki/Metal_(API)") == [
                    "https://ja.wikipedia.org/wiki/Metal_"
                ],
            misread.map { "\($0.0.debugDescription) -> \(AgentViewModel.guardURLs(in: $0.0))" }.joined(separator: " | ")
        )
        check(
            "m4g/guardNormalized",
            AgentViewModel.guardNormalized("HTTPS://Example.COM/Path/To/?Q=A#frag")
                == "https://example.com/Path/To/?Q=A"
                && AgentViewModel.guardNormalized("https://example.com/Path/") == "https://example.com/Path"
                && AgentViewModel.guardNormalized("https://example.com/") == "https://example.com"
                && AgentViewModel.guardNormalized("https://example.com/path")
                    != AgentViewModel.guardNormalized("https://example.com/Path")
                && AgentViewModel.normalized("https://Example.com/Path/") == "https://example.com/path",
            AgentViewModel.guardNormalized("HTTPS://Example.COM/Path/To/?Q=A#frag"))
        func same(_ a: String, _ b: String) -> Bool {
            AgentViewModel.guardNormalized(a) == AgentViewModel.guardNormalized(b)
        }
        check(
            "m4g/guardPercentEscapes",
            same("https://ja.wikipedia.org/wiki/Metal_%28API%29", "https://ja.wikipedia.org/wiki/Metal_(API)")
                && same(
                    "https://ja.wikipedia.org/wiki/%E3%82%A2%E3%83%83%E3%83%97%E3%83%AB",
                    "https://ja.wikipedia.org/wiki/\u{30A2}\u{30C3}\u{30D7}\u{30EB}")
                && same("https://example.com/s?q=%e3%82%a2", "https://example.com/s?q=\u{30A2}")
                && !same("https://example.com/a%2Fb", "https://example.com/a/b")
                && same("https://example.com/a%2fb", "https://example.com/a%2Fb")
                && !same("https://example.com/a%3Fb", "https://example.com/a?b")
                && !same("https://example.com/a%23b", "https://example.com/a")
                && AgentViewModel.guardNormalized("https://example.com/100%25") == "https://example.com/100%25"
                && AgentViewModel.guardNormalized("https://example.com/%E3%82x%zz") == "https://example.com/%E3%82x%zz",
            AgentViewModel.guardNormalized("https://ja.wikipedia.org/wiki/Metal_%28API%29") + " | "
                + AgentViewModel.guardNormalized("https://example.com/%E3%82x%zz"))

        func fetchCall(_ url: String) -> AgentToolCall {
            AgentToolCall(name: "fetch_page", arguments: [AgentArgument(name: "url", value: url)])
        }
        func writeCall(_ title: String, _ content: String) -> AgentToolCall {
            AgentToolCall(
                name: "write_note",
                arguments: [AgentArgument(name: "title", value: title), AgentArgument(name: "content", value: content)])
        }
        let page = AgentToolOutput(
            text: "page text", seen: ["https://example.com/old": "full page text"], finalURL: "https://example.com/new")
        let record = AgentViewModel.fetchRecord(fetchCall("https://example.com/old"), page)
        check(
            "m4g/ledgerOnlyFetchedPages",
            AgentViewModel.fetchRecord(
                AgentToolCall(name: "web_search", arguments: [AgentArgument(name: "query", value: "q")]),
                AgentToolOutput(
                    text: "1. A\n   https://example.com/snippet\n   s", seen: ["https://example.com/snippet": "A\ns"]))
                == nil
                && AgentViewModel.fetchRecord(
                    fetchCall("https://example.com/gone"),
                    AgentToolOutput(text: "error: HTTP 404 for https://example.com/gone")) == nil
                && AgentViewModel.fetchRecord(
                    fetchCall("https://example.com/empty"),
                    AgentToolOutput(text: "", finalURL: "https://example.com/empty")) == nil
                && record
                    == FetchRecord(
                        url: "https://example.com/old", finalURL: "https://example.com/new",
                        sha256: sha256("page text"), characters: 9),
            String(describing: record))
        let ledger = [record].compactMap { $0 }
        func verdict(_ answer: String, notes: [String] = [], required: Bool? = true, web: Bool = true) -> LedgerCheck {
            AgentViewModel.ledgerCheck(
                stage: "t", answer: answer, notes: notes, ledger: ledger, sourcesRequired: required, usedWebTools: web)
        }
        let redirect = verdict("A.\n\nSources:\nhttps://example.com/new\nhttps://Example.com/old/#top")
        check(
            "m4g/ledgerRedirectMatches",
            !redirect.violation && redirect.citationsTotal == 2 && redirect.citationsFetched == 2, "\(redirect)")
        let unfetched = verdict("A.\n\nSources:\nhttps://example.com/new\nhttps://example.com/snippet")
        let missing = verdict("A without sources.")
        let inBody = verdict("Per https://example.org/body-link, A.\n\nSources:\nhttps://example.com/new")
        let inNote = verdict(
            "A.\n\nSources:\nhttps://example.com/new", notes: ["Title\nsee https://example.org/note-link"])
        let passed = verdict("A.\n\nSources:\nhttps://example.com/new", notes: ["Title\nfrom https://example.com/old"])
        let autoNoWeb = verdict("A.", required: nil, web: false)
        let autoWeb = verdict("A.", required: nil, web: true)
        let notRequired = verdict("Could not read the page.", required: false)
        check(
            "m4g/guardVerdicts",
            unfetched.violation && unfetched.citationsUnfetched == ["https://example.com/snippet"]
                && unfetched.reason == "unfetched"
                && missing.violation && missing.sourcesMissing && missing.reason == "sources-missing"
                && inBody.violation && inBody.citationsUnfetched == ["https://example.org/body-link"]
                && inNote.violation && inNote.noteCitationsUnfetched == ["https://example.org/note-link"]
                && inNote.reason == "note-unfetched"
                && inNote.unfetched == ["https://example.org/note-link"]
                && !passed.violation && passed.citationsTotal == 1 && passed.citationsFetched == 1
                && !autoNoWeb.violation && !autoNoWeb.sourcesRequired && autoWeb.violation && autoWeb.sourcesMissing
                && !notRequired.violation,
            [unfetched, missing, inBody, inNote, passed, autoNoWeb, autoWeb, notRequired].map(\.logLine).joined(
                separator: " | "))
        let attempts = [
            FetchAttempt(url: "https://example.com/old", ok: true),
            FetchAttempt(url: "https://example.invalid/gone", ok: false),
        ]
        func tried(
            _ answer: String, required: Bool?, ledger: [FetchRecord] = ledger, attempts: [FetchAttempt] = attempts
        ) -> LedgerCheck {
            AgentViewModel.ledgerCheck(
                stage: "t", answer: answer, notes: [], ledger: ledger, attempts: attempts, sourcesRequired: required,
                usedWebTools: true)
        }
        let onlyFailed = [FetchAttempt(url: "https://coreai-samples-heldout.invalid/notes", ok: false)]
        let h9 = "I could not fetch https://coreai-samples-heldout.invalid/notes, so I cannot list its points."
        let reportedNo = tried(h9, required: false, ledger: [], attempts: onlyFailed)
        let reportedAuto = tried(h9, required: nil, ledger: [], attempts: onlyFailed)
        let reportedYes = tried(h9, required: true, ledger: [], attempts: onlyFailed)
        let mixed = tried(
            "A.\n\nSources:\nhttps://example.com/new\nhttps://example.invalid/gone\nhttps://example.org/never",
            required: true)
        let failedOnlyWithAPage = tried("Could not read https://example.invalid/gone.", required: nil)
        check(
            "m4g/failedFetchesAreNotViolations",
            !reportedNo.violation && reportedNo.citationsFailed == 1 && reportedNo.citationsFetched == 0
                && reportedNo.citationsUnfetched.isEmpty
                && !reportedAuto.violation && !reportedAuto.sourcesRequired
                && reportedYes.violation && reportedYes.sourcesMissing && reportedYes.reason == "sources-missing"
                && reportedYes.unfetched.isEmpty
                && AgentPromptBuilder.repairPrompt(unfetched: reportedYes.unfetched).contains(
                    "in this task: none listed. Open")
                && mixed.violation && mixed.citationsTotal == 3 && mixed.citationsFetched == 1
                && mixed.citationsFailed == 1
                && mixed.citationsUnfetched == ["https://example.org/never"] && !mixed.sourcesMissing
                && failedOnlyWithAPage.sourcesRequired && failedOnlyWithAPage.sourcesMissing,
            [reportedNo, reportedAuto, reportedYes, mixed, failedOnlyWithAPage].map(\.logLine).joined(separator: " | "))
        let percentCited = AgentViewModel.ledgerCheck(
            stage: "t", answer: "Sources:\nhttps://ja.wikipedia.org/wiki/Metal_%28API%29", notes: [],
            ledger: [
                FetchRecord(
                    url: "https://ja.wikipedia.org/wiki/Metal_(API)",
                    finalURL: "https://ja.wikipedia.org/wiki/Metal_(API)",
                    sha256: "", characters: 1)
            ],
            sourcesRequired: true, usedWebTools: true)
        check(
            "m4g/percentEscapedCitationIsFetched", !percentCited.violation && percentCited.citationsFetched == 1,
            percentCited.logLine)
        let noteOf = { (outcome: String, check: LedgerCheck) in
            AgentViewModel.guardNoteText(outcome: outcome, reason: "none", check: check, drafts: false)
        }
        let countsOnly = noteOf("verified", reportedNo)
        let labelled = noteOf("verified", passed)
        let repairedCounts = noteOf("repaired", reportedNo)
        let heldNote = AgentViewModel.guardNoteText(outcome: "held", reason: "re-violation", check: mixed, drafts: true)
        func counts(_ total: Int, _ fetched: Int, _ failed: Int) -> LedgerCheck {
            LedgerCheck(
                stage: "t", citationsTotal: total, citationsFetched: fetched, citationsFailed: failed,
                citationsUnfetched: [],
                noteCitationsUnfetched: [], sourcesRequired: true, sourcesMissing: false)
        }
        let partly = noteOf("verified", counts(4, 3, 1))
        let allThree = noteOf("verified", counts(3, 3, 0))
        let repairedAll = noteOf("repaired", counts(3, 3, 0))
        let repairedPartly = noteOf("repaired", counts(4, 3, 1))
        check(
            "m4g/passLabelNeedsAFetchedPage",
            countsOnly?.text == "0/1 cited URLs were fetched in this task · 1 could not be fetched"
                && countsOnly?.warning == false
                && partly?.text == "3/4 cited URLs were fetched in this task · 1 could not be fetched"
                && partly?.warning == false
                && allThree?.text == "\(AgentViewModel.verifiedLabel) · 3/3 cited URLs were fetched in this task"
                && repairedAll?.text
                    == "\(AgentViewModel.verifiedLabel) · after one repair turn · 3/3 cited URLs were fetched in this task"
                && repairedPartly?.text
                    == "after one repair turn · 3/4 cited URLs were fetched in this task · 1 could not be fetched"
                && labelled?.text == "\(AgentViewModel.verifiedLabel) · 1/1 cited URLs were fetched in this task"
                && repairedCounts?.text
                    == "after one repair turn · 0/1 cited URLs were fetched in this task · 1 could not be fetched"
                && heldNote?.text.hasPrefix(
                    "\(AgentViewModel.heldLabel) · re-violation · not fetched: https://example.org/never") == true
                && heldNote?.warning == true
                && AgentViewModel.guardNoteText(outcome: "skipped", reason: "cap", check: nil, drafts: false) == nil,
            [countsOnly, labelled, repairedCounts, heldNote, partly, allThree, repairedAll, repairedPartly].map {
                $0?.text ?? "nil"
            }
            .joined(separator: " | "))
        check(
            "e2e/heldVerdict",
            e2eOutcome(
                stoppedByDeadline: true, failureTitles: ["Stopped"], answer: "done", finishReason: "eos", held: true)
                == "held"
                && e2eOutcome(
                    stoppedByDeadline: false, failureTitles: [], answer: "done", finishReason: "eos", held: false)
                    == "answered",
            "")
        let now = Date()
        check(
            "m4g/repairBudgetBlockers",
            AgentViewModel.repairBlocker(stepsUsed: 8, maxSteps: 8, deadline: nil, now: now, stopped: false)
                == "no-steps-left"
                && AgentViewModel.repairBlocker(
                    stepsUsed: 4, maxSteps: 8, deadline: now.addingTimeInterval(-1), now: now, stopped: false)
                    == "deadline"
                && AgentViewModel.repairBlocker(
                    stepsUsed: 7, maxSteps: 8, deadline: now.addingTimeInterval(60), now: now, stopped: false) == nil
                && AgentViewModel.repairBlocker(stepsUsed: 4, maxSteps: 8, deadline: nil, now: now, stopped: true)
                    == "stopped",
            "")

        let folder = FileManager.default.temporaryDirectory
            .appending(path: "agent-selftest-m4g-\(UUID().uuidString)", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        func exists(_ url: URL) -> Bool { FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) }
        func lines(_ entries: [ToolCassette.Entry]) -> Data {
            let encoder = JSONEncoder()
            return Data(
                entries.compactMap { try? encoder.encode($0) }.map { String(decoding: $0, as: UTF8.self) + "\n" }
                    .joined().utf8)
        }
        let defaultsName = "agent-defaults-" + UUID().uuidString
        if let defaults = UserDefaults(suiteName: defaultsName) {
            defer { defaults.removePersistentDomain(forName: defaultsName) }
            var flagMismatches: [String] = []
            func checkFlags(over saved: Bool?) {
                for value in ["on", "off"] {
                    let agent = AgentViewModel(defaults: defaults)
                    agent.persistsSettings = false
                    _ = try? applyM4GOptions(to: agent, arguments: ["--citation-guard", value])
                    let stored = defaults.object(forKey: AgentViewModel.citationGuardDefaultsKey) as? Bool
                    if agent.citationGuard != (value == "on") || stored != saved {
                        flagMismatches.append(
                            "\(value) over \(saved.map { "saved \($0)" } ?? "no setting"): "
                                + "guard \(agent.citationGuard), stored \(stored.map { "\($0)" } ?? "nil")")
                    }
                }
            }
            let fresh = AgentViewModel(defaults: defaults)
            check(
                "m4h/defaultInstructionsAndGuard",
                fresh.citeFetched && !fresh.citationGuard
                    && defaults.object(forKey: AgentViewModel.citationGuardDefaultsKey) == nil,
                "fresh settings: citeFetched=\(fresh.citeFetched) citationGuard=\(fresh.citationGuard)")
            checkFlags(over: nil)
            fresh.citationGuard = true
            let savedOn = AgentViewModel(defaults: defaults).citationGuard
            checkFlags(over: true)
            fresh.citeFetched = false
            fresh.citationGuard = false
            let saved = AgentViewModel(defaults: defaults)
            checkFlags(over: false)
            check(
                "m4h/savedInstructionsAndGuard", savedOn && !saved.citeFetched && !saved.citationGuard,
                "saved on read back as \(savedOn), then saved off")
            check(
                "e2e/citationGuardFlagOverridesSettings", flagMismatches.isEmpty,
                flagMismatches.joined(separator: "; "))
        } else {
            check("m4h/defaultInstructionsAndGuard", false, "could not create isolated defaults")
        }
        let key = ["fetch_page", "https://example.com/old", ""]
        do {
            let tape = folder.appending(path: "cassette.jsonl")
            let tally = CallCount()
            let recorder = try ToolCassette(file: tape, mode: .record, replaysLatency: false)
            let recorded = await recorder.output(for: key) {
                await tally.add()
                try? await Task.sleep(for: .milliseconds(120))
                return page
            }
            let replayer = try ToolCassette(file: tape, mode: .replay, replaysLatency: false)
            let replayed = await replayer.output(for: key) {
                await tally.add()
                return AgentToolOutput(text: "fetched again")
            }
            let missed = await replayer.output(for: ["fetch_page", "https://example.com/other", ""]) {
                await tally.add()
                return page
            }
            let (records, hits, misses, liveCalls) = (
                await recorder.records, await replayer.hits, await replayer.misses, await tally.value
            )
            let saved = (try? String(contentsOf: tape, encoding: .utf8)) ?? ""
            check(
                "m4g/cassetteRecordReplay",
                recorded.text == "page text" && replayed.text == recorded.text && replayed.seen == page.seen
                    && replayed.finalURL == page.finalURL && records == 1 && hits == 1 && liveCalls == 1
                    && saved.contains("\"ms\":") && saved.contains("\"ok\":true"),
                "records=\(records) hits=\(hits) live=\(liveCalls) saved=\(saved.prefix(300))")
            check("m4g/cassetteReplayMiss", missed.text == ToolCassette.missText && misses == 1, missed.text)

            let wildcardTape = folder.appending(path: "wildcard.jsonl")
            let wildcardKey = ["fetch_page", "https://example.com/wild", "*"]
            let wildcardEntry = ToolCassette.Entry(
                key: wildcardKey, text: "wildcard", seen: [:], ok: true, finalURL: wildcardKey[1], ms: 0,
                recorded: "test")
            var exactEntry = wildcardEntry
            exactEntry.key[2] = "exact"
            exactEntry.text = "exact"
            var searchEntry = wildcardEntry
            searchEntry.key = ["web_search", "*", "5"]
            try lines([wildcardEntry, exactEntry, searchEntry]).write(to: wildcardTape)
            let wildcardPlayer = try ToolCassette(file: wildcardTape, mode: .replay, replaysLatency: false)
            let wildcardCalls = CallCount()
            let wildcardResult = await wildcardPlayer.output(for: ["fetch_page", wildcardKey[1], "another query"]) {
                await wildcardCalls.add()
                return AgentToolOutput(text: "unexpected live")
            }
            let exactResult = await wildcardPlayer.output(for: exactEntry.key) {
                AgentToolOutput(text: "unexpected live")
            }
            let otherURL = await wildcardPlayer.output(for: [
                "fetch_page", "https://example.com/other", "another query",
            ]) { AgentToolOutput(text: "unexpected live") }
            let otherTool = await wildcardPlayer.output(for: ["web_search", "another query", "5"]) {
                AgentToolOutput(text: "unexpected live")
            }
            let wildcardLive = await wildcardCalls.value
            let wildcardHits = await wildcardPlayer.hits
            let wildcardMisses = await wildcardPlayer.misses
            check(
                "m4h/cassetteWildcardMatchesQuery", wildcardResult.text == "wildcard" && wildcardLive == 0,
                wildcardResult.text)
            check("m4h/cassetteExactBeatsWildcard", exactResult.text == "exact", exactResult.text)
            check(
                "m4h/cassetteWildcardStaysWithURLAndTool",
                otherURL.text == ToolCassette.missText && otherTool.text == ToolCassette.missText && wildcardHits == 2
                    && wildcardMisses == 2, "hits=\(wildcardHits) misses=\(wildcardMisses)")
            for mode in [ToolCassette.Mode.record, .replayOrRecord] {
                let outputTape = folder.appending(path: "wildcard-write-" + mode.rawValue + ".jsonl")
                let writer = try ToolCassette(file: outputTape, mode: mode, replaysLatency: false)
                let reserved = await writer.output(for: wildcardKey) { page }
                let ordinaryKey = ["fetch_page", wildcardKey[1], "ordinary query"]
                _ = await writer.output(for: ordinaryKey) { page }
                let stored = try String(contentsOf: outputTape, encoding: .utf8).split(separator: "\n").map {
                    try JSONDecoder().decode(ToolCassette.Entry.self, from: Data($0.utf8))
                }
                let recordCount = await writer.records
                check(
                    "m4h/cassetteNeverRecordsWildcard-" + mode.rawValue,
                    reserved.text == page.text && stored.map(\.key) == [ordinaryKey] && recordCount == 1,
                    "records=\(recordCount) keys=\(stored.map(\.key))")
            }

            let doubled = folder.appending(path: "doubled.jsonl")
            let first = ToolCassette.Entry(
                key: key, text: "first", seen: [:], ok: true, finalURL: "https://example.com/old", ms: 300,
                recorded: "t1")
            var second = first
            second.text = "second"
            second.ms = 0
            try lines([first, second]).write(to: doubled)
            let clock = ContinuousClock()
            let slow = try ToolCassette(file: doubled, mode: .replay, replaysLatency: true)
            var mark = clock.now
            let firstWins = await slow.output(for: key) { AgentToolOutput(text: "live") }
            let waited = (clock.now - mark) / .seconds(1)
            let fast = try ToolCassette(file: doubled, mode: .replay, replaysLatency: false)
            mark = clock.now
            let quick = await fast.output(for: key) { AgentToolOutput(text: "live") }
            let skipped = (clock.now - mark) / .seconds(1)
            check(
                "m4g/cassetteFirstRecordAndLatency",
                firstWins.text == "first" && quick.text == "first" && waited >= 0.3 && skipped < 0.1,
                String(format: "first=%@ waited=%.3fs none=%.3fs", firstWins.text, waited, skipped))

            let stoppedTape = folder.appending(path: "stopped.jsonl")
            let stopping = try ToolCassette(file: stoppedTape, mode: .replayOrRecord, replaysLatency: false)
            let call = Task {
                await stopping.output(for: key) {
                    try? await Task.sleep(for: .seconds(5))
                    return AgentToolOutput(text: "error: fetch_page failed (CancellationError())")
                }
            }
            call.cancel()
            _ = await call.value
            let stoppedRecords = await stopping.records
            check(
                "m4g/cassetteSkipsStoppedCalls", stoppedRecords == 0 && !exists(stoppedTape),
                "records=\(stoppedRecords)")

            let wired = try ToolCassette(file: tape, mode: .replay, replaysLatency: false)
            var settings = AgentToolSettings(notesDirectory: folder)
            settings.cassette = wired
            let viaRun = await AgentTools.run(fetchCall("https://example.com/old"), settings: settings)
            let plainNote = await AgentTools.run(writeCall("Wired", "body"), settings: settings)
            settings.stableNotePaths = true
            let stableNote = await AgentTools.run(writeCall("Stable", "body"), settings: settings)
            let (wiredHits, wiredMisses) = (await wired.hits, await wired.misses)
            check(
                "m4g/cassetteAtToolEntry",
                viaRun.text == "page text" && viaRun.finalURL == "https://example.com/new" && wiredHits == 1
                    && wiredMisses == 0
                    && plainNote.text.hasPrefix("saved to ") && exists(folder.appending(path: "Wired.md")),
                "\(viaRun.text) hits=\(wiredHits) misses=\(wiredMisses) note=\(plainNote.text)")
            check(
                "m4g/stableNotePaths",
                stableNote.text == "saved to agent-notes/Stable.md" && exists(folder.appending(path: "Stable.md"))
                    && plainNote.text == "saved to " + folder.appending(path: "Wired.md").path(percentEncoded: false),
                stableNote.text + " | " + plainNote.text)

            let four = folder.appending(path: "four.jsonl")
            try lines(
                (1...4).map {
                    ToolCassette.Entry(
                        key: ["fetch_page", "https://example.com/p\($0)", ""], text: "page \($0)", seen: [:], ok: true,
                        finalURL: "https://example.com/p\($0)", ms: 0, recorded: "t")
                }
            ).write(to: four)
            let repairing = AgentViewModel()
            repairing.persistsSettings = false
            repairing.notesFolderPath = folder.path(percentEncoded: false)
            repairing.cassette = try ToolCassette(file: four, mode: .replay, replaysLatency: false)
            repairing.repairFetchLimit = AgentViewModel.repairFetchLimit
            let runner = repairing.toolRunner()
            let calls = (1...4).map { fetchCall("https://example.com/p\($0)") }
            let results = await withTaskGroup(of: String.self) { group in
                for call in calls { group.addTask { await runner(call) } }
                var out: [String] = []
                for await result in group { out.append(result) }
                return out
            }
            let fourHits = await repairing.cassette?.hits ?? -1
            check(
                "m4g/repairFetchLimitInParallel",
                results.count { $0 == AgentViewModel.repairFetchLimitText } == 1
                    && results.count { $0.hasPrefix("page ") } == 3
                    && repairing.repairFetches == 3 && fourHits == 3 && repairing.ledger.count == 3
                    && repairing.fetchAttempts.count == 4 && repairing.fetchAttempts.count { $0.ok } == 3,
                "\(results) reserved=\(repairing.repairFetches) hits=\(fourHits) attempts=\(repairing.fetchAttempts)")

            let mixedTape = folder.appending(path: "mixed.jsonl")
            try lines([
                ToolCassette.Entry(
                    key: ["fetch_page", "https://example.invalid/gone", ""],
                    text: "error: HTTP 404 for https://example.invalid/gone",
                    seen: [:], ok: false, finalURL: nil, ms: 0, recorded: "t"),
                ToolCassette.Entry(
                    key: ["fetch_page", "https://example.com/p1", ""], text: "page 1", seen: [:], ok: true,
                    finalURL: "https://example.com/p1", ms: 0, recorded: "t"),
            ]).write(to: mixedTape)
            let trying = AgentViewModel()
            trying.persistsSettings = false
            trying.notesFolderPath = folder.path(percentEncoded: false)
            trying.cassette = try ToolCassette(file: mixedTape, mode: .replay, replaysLatency: false)
            let tryRunner = trying.toolRunner()
            _ = await tryRunner(fetchCall("https://example.invalid/gone"))
            _ = await tryRunner(fetchCall("https://example.com/p1"))
            check(
                "m4g/fetchAttemptsRecordFailures",
                trying.fetchAttempts == [
                    FetchAttempt(url: "https://example.invalid/gone", ok: false),
                    FetchAttempt(url: "https://example.com/p1", ok: true),
                ] && trying.ledger.map(\.url) == ["https://example.com/p1"],
                "attempts=\(trying.fetchAttempts) ledger=\(trying.ledger.map(\.url))")

            let notesA = folder.appending(path: "notes-a", directoryHint: .isDirectory)
            let drafting = AgentViewModel()
            drafting.persistsSettings = false
            drafting.notesFolderPath = notesA.path(percentEncoded: false)
            drafting.guardActive = true
            let draftRunner = drafting.toolRunner()
            let saved1 = await draftRunner(writeCall("Summary", "first body"))
            let saved2 = await draftRunner(writeCall("Summary", "second body"))
            drafting.repairFetchLimit = AgentViewModel.repairFetchLimit
            let saved3 = await draftRunner(writeCall("Summary", "repaired body"))
            drafting.repairFetchLimit = nil
            let draftFolder = notesA.appending(path: AgentTools.draftFolderName, directoryHint: .isDirectory)
            let draftsBefore =
                ((try? FileManager.default.contentsOfDirectory(atPath: draftFolder.path(percentEncoded: false))) ?? [])
                .sorted()
            let committedBefore =
                ((try? FileManager.default.contentsOfDirectory(atPath: notesA.path(percentEncoded: false))) ?? [])
                .filter { $0 != AgentTools.draftFolderName }
            drafting.finishNotes(commit: true)
            let kept = (try? String(contentsOf: notesA.appending(path: "Summary 2.md"), encoding: .utf8)) ?? ""
            check(
                "m4g/draftsCommitOnPass",
                saved1 == "saved to " + notesA.appending(path: "Summary.md").path(percentEncoded: false)
                    && saved2 == "saved to " + notesA.appending(path: "Summary 2.md").path(percentEncoded: false)
                    && saved3 == saved2
                    && draftsBefore == ["Summary 2.md", "Summary.md"] && committedBefore.isEmpty
                    && kept.contains("repaired body") && exists(notesA.appending(path: "Summary.md"))
                    && !exists(draftFolder.appending(path: "Summary.md")),
                "\(saved1) | \(saved2) | \(saved3) drafts=\(draftsBefore) committedBefore=\(committedBefore)")

            for (name, body, shouldCommit, fetch) in [
                ("no-url", "A note without links.", true, false),
                ("fetched-url", "See https://example.com/p1", true, true),
                ("unfetched-url", "See https://example.org/untried", false, false),
            ] {
                let directory = folder.appending(path: "incomplete-" + name, directoryHint: .isDirectory)
                let incomplete = AgentViewModel()
                incomplete.persistsSettings = false
                incomplete.notesFolderPath = directory.path(percentEncoded: false)
                incomplete.guardActive = true
                if fetch {
                    incomplete.cassette = try ToolCassette(file: mixedTape, mode: .replay, replaysLatency: false)
                    _ = await incomplete.toolRunner()(fetchCall("https://example.com/p1"))
                }
                _ = await incomplete.toolRunner()(writeCall("Incomplete", body))
                incomplete.finishIncompleteNotes(reason: "step-limit")
                let committed = directory.appending(path: "Incomplete.md")
                let drafts = directory.appending(path: AgentTools.draftFolderName)
                let held = (try? String(contentsOf: drafts.appending(path: "HELD.txt"), encoding: .utf8)) ?? ""
                check(
                    "m4h/incompleteNotes-" + name,
                    incomplete.guardOutcome == "skipped" && exists(committed) == shouldCommit
                        && (shouldCommit
                            || (exists(drafts.appending(path: "Incomplete.md")) && held.contains("note-unfetched"))),
                    "outcome=\(incomplete.guardOutcome ?? "nil") committed=\(exists(committed)) held=\(held)")
            }

            let notesB = folder.appending(path: "notes-b", directoryHint: .isDirectory)
            let holding = AgentViewModel()
            holding.persistsSettings = false
            holding.notesFolderPath = notesB.path(percentEncoded: false)
            holding.guardActive = true
            _ = await holding.toolRunner()(writeCall("Held note", "body"))
            holding.finishNotes(commit: false)
            let held =
                (try? String(
                    contentsOf: notesB.appending(path: AgentTools.draftFolderName).appending(path: "HELD.txt"),
                    encoding: .utf8)) ?? ""
            check(
                "m4g/draftsHeldNotCommitted",
                !exists(notesB.appending(path: "Held note.md"))
                    && exists(notesB.appending(path: AgentTools.draftFolderName).appending(path: "Held note.md"))
                    && held.contains("reason: ") && held.contains("- Held note.md"),
                held)
        } catch {
            check("m4g/cassette", false, "\(error)")
        }

        check(
            "m4g/ragBudget",
            RAGContext.sourceTokenBudget == 3_000 && RAGSettings.budgetChoices == [3_000, 6_000]
                && RAGSettings().sourceTokenBudget == 3_000,
            "\(RAGContext.sourceTokenBudget) \(RAGSettings.budgetChoices)")

        let broken = folder.appending(path: "broken-bundle", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: broken, withIntermediateDirectories: true)
        try? Data("{}".utf8).write(to: broken.appending(path: "metadata.json"))
        let models = ModelsViewModel()
        models.open(broken)
        check(
            "m4g/modelsShowBundleError", models.message?.hasSuffix("metadata.json has no name.") == true,
            models.message ?? "nil")
    }

    static let longPageFixture: String = {
        var lines = ["Core ML \u{30C9}\u{30AD}\u{30E5}\u{30E1}\u{30F3}\u{30C8} \u{76EE}\u{6B21}"]
        lines += (1...60).map {
            "\u{7B2C}\($0)\u{7AE0} \u{3053}\u{306E}\u{7AE0}\u{3067}\u{306F}\u{4E00}\u{822C}\u{7684}\u{306A}\u{6982}\u{8981}\u{3068}\u{5C0E}\u{5165}\u{306E}\u{624B}\u{9806}\u{3092}\u{8AAC}\u{660E}\u{3057}\u{307E}\u{3059}\u{3002}\u{7279}\u{5225}\u{306A}\u{8A2D}\u{5B9A}\u{306F}\u{4E0D}\u{8981}\u{3067}\u{3059}\u{3002}"
        }
        lines.append(
            "\u{4ED8}\u{9332}B \u{30D0}\u{30C3}\u{30C1}\u{63A8}\u{8AD6}\u{306E}\u{30BF}\u{30A4}\u{30E0}\u{30A2}\u{30A6}\u{30C8}\u{306F}\u{65E2}\u{5B9A}\u{3067} 45 \u{79D2}\u{3067}\u{3059}\u{3002}\u{5909}\u{66F4}\u{3059}\u{308B}\u{306B}\u{306F} timeout \u{3092}\u{8A2D}\u{5B9A}\u{3057}\u{307E}\u{3059}\u{3002}"
        )
        lines += (61...80).map {
            "\u{7B2C}\($0)\u{7AE0} \u{88DC}\u{8DB3}\u{8CC7}\u{6599}\u{3068}\u{53C2}\u{8003}\u{6587}\u{732E}\u{306E}\u{4E00}\u{89A7}\u{3067}\u{3059}\u{3002}\u{8A73}\u{7D30}\u{306F}\u{5404}\u{30EA}\u{30F3}\u{30AF}\u{3092}\u{53C2}\u{7167}\u{3057}\u{3066}\u{304F}\u{3060}\u{3055}\u{3044}\u{3002}"
        }
        return lines.joined(separator: "\n")
    }()

    private actor CallCount {
        var value = 0
        func add() { value += 1 }
    }

    static let htmlFixture = """
        <!-- a comment with <b>tags</b> -->
        <html><head><style>body { color: red }</style></head>
        <body><script>if (1 < 2) { hide("&amp;"); }</script>
        <p>Hello &amp; welcome</p>
        <p>It&#x27;s   here</p>
        <noscript>enable javascript</noscript>
        <div>Tail &mdash; end</div>
        </body></html>
        """

    static let duckDuckGoFixture = """
        <div class="results">
          <div class="result results_links web-result">
            <h2 class="result__title">
              <a rel="nofollow" class="result__a" href="https://developer.apple.com/documentation/coreml">Core ML | Apple <b>Developer</b> Documentation</a>
            </h2>
            <a class="result__url" href="https://developer.apple.com/documentation/coreml">developer.apple.com</a>
            <a class="result__snippet" href="https://developer.apple.com/documentation/coreml">Use <b>Core</b> <b>ML</b> on a person&#x27;s device.</a>
          </div>
          <div class="result results_links web-result">
            <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fcore%2Dml&amp;rut=deadbeef">Core ML Tools</a>
            <a class="result__snippet" href="https://example.com/core-ml">Convert models &amp; run them.</a>
          </div>
          <div class="result results_links web-result">
            <a rel="nofollow" class="result__a" href="https://example.org/bare">No snippet here</a>
          </div>
          <div class="result results_links web-result">
            <a rel="nofollow" class="result__a" href="https://example.net/fourth">Fourth result</a>
            <a class="result__snippet" href="https://example.net/fourth">Fourth snippet.</a>
          </div>
        </div>
        """

    @MainActor
    private static func fetchAddressChecks(offline: Bool, _ check: (String, Bool, String) -> Void) async {
        guard let publicURL = URL(string: "https://public.example/page"),
            let privateURL = URL(string: "http://127.0.0.1/private")
        else {
            check("fetch/addressFixtures", false, "Invalid URL fixture")
            return
        }
        let mapped = [UInt8](repeating: 0, count: 10) + [255, 255, 127, 0, 0, 1]
        for (name, addresses, permitted) in [
            ("loopback", [[127, 0, 0, 1]], false),
            ("private", [[10, 0, 0, 1]], false),
            ("shared", [[100, 64, 0, 1]], false),
            ("mapped", [mapped], false),
            ("mixed", [[8, 8, 8, 8], [192, 168, 0, 1]], false),
            ("public", [[8, 8, 8, 8], [1, 1, 1, 1]], true),
            ("empty", [], false),
        ] {
            do {
                try await AgentFetchPolicy(resolve: { _ in addresses }).validate(publicURL)
                check("fetch/resolved/" + name, permitted, "Unexpectedly allowed")
            } catch {
                check("fetch/resolved/" + name, !permitted, "\(error)")
            }
        }
        for address in ["2130706433", "0x7f.1", "0177.0.0.1", "[::ffff:7f00:1]", "localhost."] {
            do {
                try await AgentFetchPolicy().validate(URL(string: "http://\(address)/"))
                check("fetch/numeric/" + address, false, "Unexpectedly allowed")
            } catch {
                check("fetch/numeric/" + address, true, "\(error)")
            }
        }
        let cancelledScope = AgentAddressScope()
        cancelledScope.cancel()
        for name in ["8.8.8.8", "public.example"] {
            do {
                _ = try await AgentAddressResolver.resolve(name, scope: cancelledScope)
                check("fetch/cancelledResolutionScope/" + name, false, "Unexpected success")
            } catch is CancellationError {
                check("fetch/cancelledResolutionScope/" + name, true, "")
            } catch { check("fetch/cancelledResolutionScope/" + name, false, "\(error)") }
        }
        let policy = AgentFetchPolicy(resolve: { host in
            host == "public.example" ? [[8, 8, 8, 8]] : [[127, 0, 0, 1]]
        })
        do {
            let page = try await RAGRetriever.readPage(
                url: publicURL, policy: policy,
                download: { _ in throw AgentToolMessage(text: "error: HTTP 403", fatal: true) },
                render: { _ in "Public page rendered after a direct HTTP failure." })
            check(
                "fetch/ragKeepsRenderFallback", page.text == "Public page rendered after a direct HTTP failure.",
                page.text)
        } catch { check("fetch/ragKeepsRenderFallback", false, "\(error)") }
        do {
            _ = try await RAGRetriever.readPage(
                url: publicURL, policy: policy,
                download: { _ in
                    throw AgentToolMessage(text: AgentTools.localAddressRefusal, fatal: true, forbidsPageFallback: true)
                },
                render: { _ in throw AgentToolMessage(text: "error: forbidden render started") })
            check("fetch/ragRefusalSkipsRender", false, "Unexpected success")
        } catch let error as AgentToolMessage {
            check("fetch/ragRefusalSkipsRender", error.text == AgentTools.localAddressRefusal, error.text)
        } catch { check("fetch/ragRefusalSkipsRender", false, "\(error)") }
        if let response = HTTPURLResponse(url: publicURL, statusCode: 302, httpVersion: nil, headerFields: nil) {
            let redirects = AgentFetchRedirects(policy: policy)
            let task = URLSession.shared.dataTask(with: publicURL)
            let next = await redirects.urlSession(
                .shared, task: task, willPerformHTTPRedirection: response, newRequest: URLRequest(url: privateURL))
            var text = "redirect body"
            do { try redirects.checkResponse() } catch let error as AgentToolMessage { text = error.text } catch {}
            check("fetch/redirectBeforeRequest", next == nil && text == AgentTools.localAddressRefusal, text)
        }
        for mainFrame in [true, false] {
            let reader = AgentPageReader(policy: policy)
            let decision = await reader.navigationPolicy(url: privateURL, isMainFrame: mainFrame)
            check("fetch/webNavigation/\(mainFrame ? "main" : "subframe")", decision == .cancel, "\(decision)")
        }
        let reader = AgentPageReader(policy: policy)
        do {
            _ = try await reader.configuration()
            check("fetch/contentRulesInstalled", reader.contentRulesInstalled, "Rules were not installed")
        } catch { check("fetch/contentRulesInstalled", false, "\(error)") }
        for evaluation in [false, true] {
            for outcome in ["refusal", "cancellation", "success", "before-registration"] {
                let waiting = AgentPageReader(policy: policy)
                let refusal = AgentToolMessage(text: AgentTools.localAddressRefusal, fatal: true)
                if outcome == "before-registration" { waiting.abort(with: refusal) }
                var result = "success"
                do {
                    if evaluation {
                        _ = try await waiting.waitForEvaluation { complete in
                            if outcome == "refusal" { waiting.abort(with: refusal) }
                            if outcome == "cancellation" { waiting.abort(with: CancellationError()) }
                            complete(.success("page"))
                            complete(.failure(refusal))
                            waiting.abort(with: CancellationError())
                        }
                    } else {
                        try await waiting.waitForNavigation {
                            if outcome == "refusal" { waiting.abort(with: refusal) }
                            if outcome == "cancellation" { waiting.abort(with: CancellationError()) }
                            waiting.finish(.success(()))
                            waiting.finish(.failure(refusal))
                            waiting.abort(with: CancellationError())
                        }
                    }
                } catch let error as AgentToolMessage { result = error.text } catch is CancellationError {
                    result = "cancellation"
                } catch { result = "\(error)" }
                let expected = outcome == "refusal" || outcome == "before-registration" ? refusal.text : outcome
                check("fetch/continuation/\(evaluation)/\(outcome)", result == expected, result)
            }
        }
        if !offline, let raw = LaunchArguments.value("--fetch-redirect-test-url"), let url = URL(string: raw) {
            let output = await AgentTools.run(
                AgentToolCall(name: "fetch_page", arguments: [AgentArgument(name: "url", value: raw)]),
                settings: AgentToolSettings())
            check("fetch/liveRedirect/agent", output.text == AgentTools.localAddressRefusal, output.text)
            do {
                _ = try await RAGRetriever.readPage(url: url)
                check("fetch/liveRedirect/rag", false, "Unexpectedly allowed")
            } catch let error as AgentToolMessage {
                check("fetch/liveRedirect/rag", error.text == AgentTools.localAddressRefusal, error.text)
            } catch { check("fetch/liveRedirect/rag", false, "\(error)") }
        }
    }

    @MainActor
    private static func captureFetchPages(_ directory: String, _ check: (String, Bool, String) -> Void) async {
        let addresses = [
            "https://developer.apple.com/documentation/coreml",
            "https://en.wikipedia.org/wiki/Metal_(API)",
            "https://www.apple.com/newsroom/2025/02/apple-debuts-iphone-16e-a-powerful-new-member-of-the-iphone-16-family/",
        ]
        let destination = URL(filePath: directory, directoryHint: .isDirectory)
        do { try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true) } catch {
            check("fetch/captureDirectory", false, "\(error)")
            return
        }
        for (index, raw) in addresses.enumerated() {
            let output = await AgentTools.run(
                AgentToolCall(name: "fetch_page", arguments: [AgentArgument(name: "url", value: raw)]),
                settings: AgentToolSettings(pageCharacterBudget: 1_000_000))
            do {
                try output.text.write(
                    to: destination.appending(path: "page-\(index).txt"), atomically: true, encoding: .utf8)
                check(
                    "fetch/capture/\(index)", output.finalURL != nil && output.text.count > 400,
                    "\(output.text.prefix(120))")
            } catch { check("fetch/capture/\(index)", false, "\(error)") }
        }
    }
}
