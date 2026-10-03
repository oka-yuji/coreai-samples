import Foundation
import FoundationModels

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
            let code = await executeEndToEnd(effort: effort, deadline: LaunchArguments.value("--deadline").flatMap { Double($0) } ?? 900)
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
        let chat = ChatViewModel()
        out("[e2e] loading \(model)")
        let loadStart = Date()
        await chat.load(URL(filePath: model, directoryHint: .isDirectory))
        guard chat.phase == .ready, let stats = chat.loadStats else {
            out("agent-e2e: load failed — \(chat.phase)")
            return 1
        }
        out(String(format: "[e2e] loaded %@ in %.1fs", chat.modelName, Date().timeIntervalSince(loadStart)))
        out(String(
            format: "[e2e] prepared: cached=%@ specializeSeconds=%.2f contextLimit=%@", String(stats.cached), stats.specializeSeconds,
            chat.contextLimit.map(String.init) ?? "default"))
        let agent = AgentViewModel()
        agent.persistsSettings = false
        agent.effort = effort
        if let steps = LaunchArguments.value("--max-steps").flatMap({ Int($0) }) { agent.maxIterations = steps }
        if let tokens = LaunchArguments.value("--max-tokens").flatMap({ Int($0) }) { agent.maxNewTokens = tokens }
        if let budget = LaunchArguments.value("--page-budget").flatMap({ Int($0) }) { agent.pageCharacterBudget = budget }
        agent.allowFileOperations = LaunchArguments.flag("--file-ops")
        if let notes = LaunchArguments.value("--notes-dir") { agent.notesFolderPath = notes }
        out("[e2e] effort=\(agent.effort.rawValue) steps=\(agent.maxIterations) tokens=\(agent.maxNewTokens) "
            + "pageBudget=\(agent.pageCharacterBudget) search=\(agent.searchProvider) date=\(AgentPromptBuilder.today()) "
            + "fileOps=\(agent.allowFileOperations) notesDir=\(agent.notesDirectory.path(percentEncoded: false))")
        out("[e2e] task: \(task)")
        agent.input = task
        let started = Date()
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
        out(String(
            format: "[e2e] %.1fs  toolCalls=%d  entries=%d  failures=%d", Date().timeIntervalSince(started),
            agent.entries.filter { $0.kind == .toolCall }.count, agent.entries.count, failureTitles.count))
        out("[e2e] status: \(agent.statusLine)")
        let answer = agent.entries.last { $0.kind == .answer }?.text ?? ""
        out("[e2e] final answer (\(answer.count) chars):\n\(answer)")
        let outcome = e2eOutcome(
            stoppedByDeadline: stoppedByDeadline, failureTitles: failureTitles, answer: answer, finishReason: agent.lastFinishReason)
        out("[e2e] outcome: \(outcome)")
        out((outcome == answeredOutcome ? "AGENT E2E COMPLETED" : "AGENT E2E INCOMPLETE")
            + "  diverges=\(agent.divergeCount)  capSteps=\(agent.capSteps)")
        return outcome == answeredOutcome ? 0 : 1
    }

    static let answeredOutcome = "answered"

    static func e2eOutcome(stoppedByDeadline: Bool, failureTitles: [String], answer: String, finishReason: String?) -> String {
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

        let golden = DateComponents(calendar: Calendar(identifier: .gregorian), year: 2026, month: 8, day: 26, hour: 12).date ?? .now
        check("prompt/today", AgentPromptBuilder.today(golden) == goldenDate, AgentPromptBuilder.today(golden))
        let instructions = AgentPromptBuilder.instructions(date: goldenDate)
        check("prompt/instructions",
            instructions.hasPrefix("You are an agent running on this device. Use the tools to gather facts before you answer.")
                && instructions.contains("answer directly without calling a tool.\n\n- If a tool returns an error")
                && instructions.components(separatedBy: "\n- ").count == 4
                && instructions.contains("end the answer with a \"Sources:\" section")
                && instructions.hasSuffix("past tense and prefer the most recent sources.\n\nCurrent date: " + goldenDate),
            instructions.debugDescription)

        let run: @Sendable (AgentToolCall) async -> String = { call in
            await AgentTools.run(call, settings: AgentToolSettings(notesDirectory: FileManager.default.temporaryDirectory)).text
        }
        let plain = AgentPromptBuilder.tools(fileOperations: false, run: run)
        let withFiles = AgentPromptBuilder.tools(fileOperations: true, run: run)
        let search = plain.first.flatMap { try? JSONSerialization.jsonObject(with: JSONEncoder().encode($0.parameters)) as? [String: Any] }
        let properties = search?["properties"] as? [String: [String: Any]]
        check("tools/definitions",
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
        check("tools/argumentsInDefinitionOrder",
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
        let selected = AgentTools.pageText(longPageFixture, query: "\u{30D0}\u{30C3}\u{30C1}\u{63A8}\u{8AD6}\u{306E}\u{30BF}\u{30A4}\u{30E0}\u{30A2}\u{30A6}\u{30C8}", budget: budget)
        check("fetch/queryFindsLatePassage",
            !head.contains("45 \u{79D2}") && selected.contains("45 \u{79D2}") && selected.contains("\u{4ED8}\u{9332}B")
            && selected.hasPrefix("Core ML \u{30C9}\u{30AD}\u{30E5}\u{30E1}\u{30F3}\u{30C8} \u{76EE}\u{6B21}") && selected.contains("\n…\n")
            && selected.count <= budget + 200,
            "head=\(head.contains("45 \u{79D2}")) selected=\(selected.count) chars: " + selected.prefix(120).debugDescription)

        check("fetch/noQueryIsUnchanged",
            AgentTools.pageText(longPageFixture, query: "", budget: budget) == head
            && AgentTools.pageText(longPageFixture, query: "   ", budget: budget) == head,
            "a page fetched without a query must still be the plain head")

        let hostileQuery = "\"\u{30D0}\u{30C3}\u{30C1}\u{63A8}\u{8AD6}\" -\u{306E} AND* (\u{30BF}\u{30A4}\u{30E0}\u{30A2}\u{30A6}\u{30C8}) col:on"
        let hostileResult = AgentTools.pageText(longPageFixture, query: hostileQuery, budget: budget)
        check("fetch/hostileQuerySurvives",
            hostileResult.contains("45 \u{79D2}")
            && AgentTools.matchTerms(hostileQuery).allSatisfy { !$0.contains("\"") }
            && AgentTools.pageText(longPageFixture, query: "?!()", budget: budget) == head,
            "terms=\(AgentTools.matchTerms(hostileQuery))")

        let overlapping = AgentViewModel.gramOverlap("abcdefghij", "zz abcdefghi zz")
        let disjoint = AgentViewModel.gramOverlap("abcdefghij", "0123456789")
        let answer = "Core ML runs on device.\n\nSources:\nhttps://example.com/read\n- https://example.org/never-opened"
        let split = AgentViewModel.splitSources(answer)
        check("cite/overlapAndUnseenURL",
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
        check("search/duckDuckGoFixture", results == [
            AgentTools.SearchResult(
                title: "Core ML | Apple Developer Documentation", url: "https://developer.apple.com/documentation/coreml",
                snippet: "Use Core ML on a person's device."),
            AgentTools.SearchResult(title: "Core ML Tools", url: "https://example.com/core-ml", snippet: "Convert models & run them."),
            AgentTools.SearchResult(title: "No snippet here", url: "https://example.org/bare", snippet: ""),
            AgentTools.SearchResult(title: "Fourth result", url: "https://example.net/fourth", snippet: "Fourth snippet."),
        ], "\(results)")

        check("search/rateLimited", AgentTools.searchStatusError(202, brave: false)
            == "error: search backend returned HTTP 202 (rate limited or blocked). Do not retry the "
            + "same query immediately; try a different tool or rephrase once."
            && AgentTools.searchStatusError(429, brave: false)?.contains("HTTP 429") == true
            && AgentTools.searchStatusError(401, brave: true)?.contains("Brave API key rejected") == true
            && AgentTools.searchStatusError(402, brave: true)?.contains("paid plan") == true,
            AgentTools.searchStatusError(202, brave: false) ?? "nil")

        check("search/ok200", AgentTools.searchStatusError(200, brave: false) == nil
            && AgentTools.searchStatusError(200, brave: true) == nil, "200 must not be an error")

        let hostile = "../../etc/pas swd: <hack>"
        let hostileName = AgentTools.noteFileName(hostile)
        check("notes/fileName", !hostileName.contains("/") && !hostileName.contains("..")
            && hostileName.hasSuffix(".md") && AgentTools.noteFileName("   ") == "note.md"
            && AgentTools.noteFileName(String(repeating: "a", count: 300)).count == 83, hostileName)
        do {
            let path = try AgentTools.writeNote(title: hostile, content: "body")
            let inside = URL(fileURLWithPath: path).deletingLastPathComponent().standardizedFileURL
                == AgentTools.defaultNotesDirectory().standardizedFileURL
            check("notes/write", inside && FileManager.default.fileExists(atPath: path) && path.contains("/agent-notes/"), path)
            let second = try AgentTools.writeNote(title: hostile, content: "other")
            let third = try AgentTools.writeNote(title: hostile, content: "more")
            let kept = (try? String(contentsOf: URL(fileURLWithPath: path), encoding: .utf8)) ?? ""
            let base = String(path.dropLast(3))
            check("notes/writeSuffixesOnCollision",
                second == base + " 2.md" && third == base + " 3.md" && kept.contains("body") && !kept.contains("other"), second)
            try? FileManager.default.removeItem(atPath: third)
            try? FileManager.default.removeItem(atPath: second)
            try? FileManager.default.removeItem(atPath: path)
        } catch {
            check("notes/write", false, "\(error)")
            check("notes/writeSuffixesOnCollision", false, "\(error)")
        }

        let home = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true).standardizedFileURL.resolvingSymlinksInPath()
        let rootsFromHome = AgentTools.defaultFileRoots(notesDirectory: home)
        let rootsFromRoot = AgentTools.defaultFileRoots(notesDirectory: URL(fileURLWithPath: "/", isDirectory: true))
        check("notes/rootsExcludeHome",
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
            check("notes/move", moved.hasSuffix("/archive/Move me.md") && FileManager.default.fileExists(atPath: moved)
                && !FileManager.default.fileExists(atPath: saved), moved)
        } catch {
            check("notes/move", false, "\(error)")
        }
        _ = try? AgentTools.writeNote(title: "Move me", content: "body", directory: movesIn)
        var refusals = [
            refusal { try AgentTools.moveNote(
                title: "Move me", destination: "/tmp/agent-refuse", notesDirectory: movesIn,
                allowedRoots: [movesIn.standardizedFileURL.resolvingSymlinksInPath()]) },
            refusal { try AgentTools.moveNote(
                title: "Move me", destination: archive.path(percentEncoded: false), notesDirectory: movesIn) },
        ]
        refusals.append(await AgentTools.run(
            AgentToolCall(name: "move_note", arguments: [
                AgentArgument(name: "title", value: "Move me"), AgentArgument(name: "destination", value: "Desktop"),
            ]),
            settings: AgentToolSettings(notesDirectory: movesIn)).text)
        check("notes/moveRefused", refusals.allSatisfy { $0.hasPrefix("error:") }, refusals.joined(separator: " | "))
        try? FileManager.default.removeItem(at: movesIn)

        let capTitle = "Answer cut at the token limit (1536)"
        check("e2e/outcomeVerdict",
            e2eOutcome(stoppedByDeadline: false, failureTitles: [], answer: "done", finishReason: "eos") == "answered"
            && e2eOutcome(stoppedByDeadline: true, failureTitles: [], answer: "done", finishReason: "eos") == "stopped"
            && e2eOutcome(stoppedByDeadline: false, failureTitles: [], answer: AgentViewModel.noAnswerText, finishReason: "eos") == "failed"
            && e2eOutcome(stoppedByDeadline: false, failureTitles: [], answer: "done", finishReason: "cap") == "failed"
            && e2eOutcome(stoppedByDeadline: false, failureTitles: [], answer: "done", finishReason: nil) == "failed"
            && e2eOutcome(stoppedByDeadline: false, failureTitles: ["Step limit reached"], answer: "done", finishReason: "eos") == "step-limit"
            && e2eOutcome(stoppedByDeadline: false, failureTitles: ["Context window is full"], answer: "done", finishReason: "eos") == "overflow"
            && e2eOutcome(stoppedByDeadline: false, failureTitles: [capTitle], answer: "done", finishReason: "cap") == "cap"
            && e2eOutcome(stoppedByDeadline: false, failureTitles: ["Stopped"], answer: "", finishReason: nil) == "stopped"
            && e2eOutcome(stoppedByDeadline: false, failureTitles: ["Generation failed"], answer: "done", finishReason: "eos") == "failed",
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
        check("transcript/markdown",
            present.count == offsets.count && present == present.sorted()
                && markdown.contains("### Thinking (\(thinkText.count) chars)")
                && markdown.contains("### web_search result (\(resultText.count) chars)")
                && markdown.contains("### Calling web_search")
                && markdown.contains("\n_\(statsText)_\n")
                && markdown.contains("````"),
            "offsets \(offsets)")

        if offline {
            out("SKIP  search/live (--offline)")
            out("SKIP  html/webView (--offline)")
        } else {
            do {
                let live = try await AgentTools.duckDuckGoResults(query: "Core ML", limit: 5)
                if live.isEmpty {
                    out("SKIP  search/live (no results — network or endpoint change)")
                } else {
                    check("search/live", live.allSatisfy { !$0.title.isEmpty && $0.url.hasPrefix("http") }, "\(live.prefix(2))")
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
                    out("      webView: \(live.count) chars, first line: " + (live.split(separator: "\n").first.map(String.init) ?? ""))
                } catch {
                    out("SKIP  html/webView \(url.host() ?? address) (\(error))")
                }
            }
        }

        out(failures == 0 ? "AGENT SELFTEST PASS" : "AGENT SELFTEST FAIL (\(failures))")
        return failures == 0 ? 0 : 1
    }

    static let goldenDate = "2026-08-26 (Wed)"

    static let longPageFixture: String = {
        var lines = ["Core ML \u{30C9}\u{30AD}\u{30E5}\u{30E1}\u{30F3}\u{30C8} \u{76EE}\u{6B21}"]
        lines += (1...60).map {
            "\u{7B2C}\($0)\u{7AE0} \u{3053}\u{306E}\u{7AE0}\u{3067}\u{306F}\u{4E00}\u{822C}\u{7684}\u{306A}\u{6982}\u{8981}\u{3068}\u{5C0E}\u{5165}\u{306E}\u{624B}\u{9806}\u{3092}\u{8AAC}\u{660E}\u{3057}\u{307E}\u{3059}\u{3002}\u{7279}\u{5225}\u{306A}\u{8A2D}\u{5B9A}\u{306F}\u{4E0D}\u{8981}\u{3067}\u{3059}\u{3002}"
        }
        lines.append("\u{4ED8}\u{9332}B \u{30D0}\u{30C3}\u{30C1}\u{63A8}\u{8AD6}\u{306E}\u{30BF}\u{30A4}\u{30E0}\u{30A2}\u{30A6}\u{30C8}\u{306F}\u{65E2}\u{5B9A}\u{3067} 45 \u{79D2}\u{3067}\u{3059}\u{3002}\u{5909}\u{66F4}\u{3059}\u{308B}\u{306B}\u{306F} timeout \u{3092}\u{8A2D}\u{5B9A}\u{3057}\u{307E}\u{3059}\u{3002}")
        lines += (61...80).map { "\u{7B2C}\($0)\u{7AE0} \u{88DC}\u{8DB3}\u{8CC7}\u{6599}\u{3068}\u{53C2}\u{8003}\u{6587}\u{732E}\u{306E}\u{4E00}\u{89A7}\u{3067}\u{3059}\u{3002}\u{8A73}\u{7D30}\u{306F}\u{5404}\u{30EA}\u{30F3}\u{30AF}\u{3092}\u{53C2}\u{7167}\u{3057}\u{3066}\u{304F}\u{3060}\u{3055}\u{3044}\u{3002}" }
        return lines.joined(separator: "\n")
    }()

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
}
