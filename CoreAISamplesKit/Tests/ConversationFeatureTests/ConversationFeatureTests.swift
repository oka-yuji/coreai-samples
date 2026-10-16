import ConversationFeature
import FeatureStore
import Foundation
import Testing

typealias Commands = [ConversationFeature.Command]

let launch = Date(timeIntervalSinceReferenceDate: 813_456_000)

func reduce(_ state: inout ConversationState, _ actions: ConversationAction...) -> [Commands] {
    actions.map { ConversationFeature.reduce(&state, $0) }
}

func started(
    _ mode: ConversationMode, id: String = "c1", settings: ConversationSettings = ConversationSettings(),
    prompt: String = "Question"
) -> ConversationState {
    var state = ConversationState(id: id, now: launch, settings: settings)
    _ = ConversationFeature.reduce(&state, .send(prompt, mode: mode, now: launch))
    return state
}

func tool(_ id: String, _ name: String, _ arguments: String) -> ToolCallRecord {
    ToolCallRecord(id: id, name: name, arguments: arguments)
}

func calls(_ id: String, _ calls: ToolCallRecord...) -> TranscriptRecord.Entry {
    .toolCalls(id: id, calls: calls, metadata: StepMetadata(stopReason: "stopToken"))
}

func output(_ call: ToolCallRecord, _ text: String) -> TranscriptRecord.Entry {
    .toolOutput(id: call.id, name: call.name, text: text)
}

func answer(_ id: String, _ text: String, stop: String = "stopToken") -> TranscriptRecord.Entry {
    .response(id: id, text: text, metadata: StepMetadata(stopReason: stop))
}

func turn(_ prompt: String, _ entries: [TranscriptRecord.Entry]) -> TranscriptRecord {
    TranscriptRecord(entries: [.instructions(id: "i", text: "Rules"), .prompt(id: "p", text: prompt)] + entries)
}

func stats(_ step: Int, calls: Int = 0, finish: String = "eos") -> StepStats {
    StepStats(
        step: step, ttftSeconds: 0.5, tokensPerSecond: 20, promptTokens: 100 * step, reusedTokens: 50,
        generatedTokens: 30, toolCalls: calls, finish: finish)
}

func snapshot(_ commands: Commands) -> ConversationSnapshot? {
    guard case .run(.save(let snapshot), .save)? = commands.first else { return nil }
    return snapshot
}

struct ConversationFeatureTests {
    @Test("Chat mode allows web_search and fetch_page and refuses write_note and move_note with a reason")
    func chatAllowsLookupsAndRefusesSaving() {
        let policy = TurnPolicy(.chat, settings: ConversationSettings())
        #expect(policy.allowedTools == ["web_search", "fetch_page"])
        #expect(["web_search", "fetch_page"].map(policy.refusal(for:)) == [nil, nil])
        #expect(policy.refusal(for: "write_note") == "error: saving notes needs Work mode")
        #expect(policy.refusal(for: "append_note") == "error: saving notes needs Work mode")
        #expect(policy.refusal(for: "move_note") == "error: saving notes needs Work mode")

        var state = started(.chat, settings: ConversationSettings(chatMaxSteps: 2))
        let search = tool("t1", "web_search", #"{"query":"swift"}"#)
        let fetch = tool("t2", "fetch_page", #"{"url":"https://swift.org"}"#)
        let allowed = reduce(&state, .toolRequested(run: 1, search), .toolRequested(run: 1, fetch))
        #expect(
            allowed == [
                [.run(.runTool(search, finalize: true), id: nil)], [.run(.runTool(fetch, finalize: true), id: nil)],
            ])
        #expect(!state.refusedSave)

        let note = tool("t3", "write_note", #"{"title":"Plan","content":"Draft"}"#)
        let move = tool("t4", "move_note", #"{"title":"Plan","destination":"Desktop"}"#)
        let reason = ConversationFeature.finalized("error: saving notes needs Work mode")
        let refused = reduce(&state, .toolRequested(run: 1, note), .toolRequested(run: 1, move))
        #expect(
            refused == [
                [.run(.refuseTool(note, reason: reason), id: nil)], [.run(.refuseTool(move, reason: reason), id: nil)],
            ])
        #expect(state.refusedSave && state.isRunning && state.finalization == .normal)
    }

    @Test("Work mode allows every tool and keeps the step limit, the token limit and the expanded display")
    func workAllowsEveryTool() {
        let tools = ["web_search", "fetch_page", "write_note", "append_note", "move_note"]
        let work = TurnPolicy(.work, settings: ConversationSettings())
        #expect(work.allowedTools == Set(tools) && work.refusedTools.isEmpty)
        #expect(tools.allSatisfy { work.refusal(for: $0) == nil })
        #expect(work.maxSteps == 8 && work.maxTokens == 4_096 && !work.collapsesSteps)
        let chat = TurnPolicy(.chat, settings: ConversationSettings())
        #expect(chat.maxSteps == 16 && chat.maxTokens == 2_048 && chat.collapsesSteps)

        let tuned = ConversationSettings(maxSteps: 5, maxTokens: 1_536)
        let policy = TurnPolicy(.work, settings: tuned)
        #expect(policy.maxSteps == 5 && policy.maxTokens == 1_536)
        #expect(TurnPolicy(.chat, settings: tuned).maxSteps == 16)

        let note = tool("t1", "write_note", #"{"title":"Plan","content":"Draft"}"#)
        let record = turn("Save a plan", [.reasoning(id: "r", text: "Save it"), calls("c", note)])
        var state = ConversationState(id: "c1", now: launch, settings: tuned)
        let commands = reduce(
            &state, .send("Save a plan", mode: .work, now: launch), .transcriptChanged(run: 1, record),
            .toolRequested(run: 1, note))
        #expect(
            commands == [
                [.run(.generate(run: 1, prompt: "Save a plan", policy: policy, finalize: false), id: .generation)],
                [],
                [.run(.runTool(note, finalize: false), id: nil)],
            ])
        #expect(!state.refusedSave)
        #expect(state.entries.map(\.kind) == [.user, .thinking, .toolCall])
        #expect(state.entries.map(\.collapsed) == [false, false, false])

        var chatState = ConversationState(id: "c2", now: launch, settings: tuned)
        _ = reduce(&chatState, .send("Save a plan", mode: .chat, now: launch), .transcriptChanged(run: 1, record))
        #expect(chatState.entries.map(\.collapsed) == [false, true, true])
    }

    @Test("The mode chosen in the composer applies to the next send and cannot change while a turn is running")
    func modeAppliesToTheNextSend() {
        var state = ConversationState(id: "c1", now: launch)
        _ = reduce(&state, .modeChanged(.work))
        #expect(state.mode == .work && state.lastMode == nil)
        let work = TurnPolicy(.work, settings: ConversationSettings())
        let first = reduce(&state, .send("Research this", mode: .work, now: launch))
        #expect(
            first == [
                [.run(.generate(run: 1, prompt: "Research this", policy: work, finalize: false), id: .generation)]
            ])

        let running = state
        let ignored = reduce(
            &state, .modeChanged(.chat), .settingsChanged(ConversationSettings(maxSteps: 3)),
            .send("Another", mode: .chat, now: launch + 1))
        #expect(ignored == [[], [], []])
        #expect(state == running)

        _ = reduce(
            &state, .transcriptChanged(run: 1, turn("Research this", [answer("a", "Done")])),
            .generationEnded(run: 1, stopReason: "stopToken"))
        #expect(!state.isRunning && state.lastMode == .work)
        _ = reduce(&state, .modeChanged(.chat))
        let chat = TurnPolicy(.chat, settings: ConversationSettings())
        let next = reduce(&state, .send("Thanks", mode: .chat, now: launch + 60))
        #expect(next == [[.run(.generate(run: 2, prompt: "Thanks", policy: chat, finalize: false), id: .generation)]])
        #expect(state.turns.map(\.mode) == [.work, .chat] && state.lastMode == .chat)
        #expect(state.createdAt == launch && state.updatedAt == launch + 60)
    }

    @Test(
        "Steps are counted from the transcript the same way as before, and the limit cancels after the last allowed step"
    )
    func countsStepsFromTheTranscript() {
        let search = tool("t1", "web_search", #"{"query":"swift"}"#)
        let first = tool("t2", "fetch_page", #"{"url":"https://a.example/a"}"#)
        let second = tool("t3", "fetch_page", #"{"url":"https://b.example/b"}"#)
        let record = TranscriptRecord(entries: [
            .instructions(id: "i", text: "Rules"),
            .prompt(id: "p", text: "Question"),
            .reasoning(id: "r1", text: "Search first"),
            calls("c1", search),
            .response(id: "between", text: "and then", metadata: nil),
            calls("c2", first),
            output(search, "results"),
            output(first, "page a"),
            .reasoning(id: "r2", text: "One more"),
            calls("c3", second),
            output(second, "page b"),
            answer("a", "Done"),
        ])
        #expect(
            record.steps(from: 2) == [
                TranscriptStep(generated: 2..<6, outputs: 6..<8),
                TranscriptStep(generated: 8..<10, outputs: 10..<11),
                TranscriptStep(generated: 11..<12, outputs: nil),
            ])
        #expect(record.steps(from: 6).first == TranscriptStep(generated: 6..<6, outputs: 6..<8))
        #expect(record.steps(from: 12).isEmpty && record.steps(from: 2, to: 6).count == 1)

        var state = started(.work, settings: ConversationSettings(maxSteps: 2))
        func observe(_ count: Int) -> Commands {
            ConversationFeature.reduce(
                &state, .transcriptChanged(run: 1, TranscriptRecord(entries: Array(record.entries.prefix(count)))))
        }
        #expect(observe(6) == [] && state.step == 1 && state.finalization == .normal)
        #expect(observe(8) == [] && state.step == 2 && state.finalization == .lastStep)
        #expect(observe(11) == [.run(.stopForBudget(run: 1), id: .stop)])
        #expect(state.step == 3 && state.finalization == .cutOff && !state.stopRequested && state.isRunning)
        #expect(observe(12) == [])
        #expect(!state.entries.contains { $0.text == "Done" })
        #expect(state.entries.last?.title == "Ended at the step limit")
    }

    @Test(
        "One step before the limit, the finalization note is added to the tool output, and a further tool call ends the turn with the collected facts and the open items"
    )
    func finalizesOneStepBeforeTheLimit() throws {
        var state = started(.work, settings: ConversationSettings(maxSteps: 3), prompt: "Compare the pages")
        let search = tool("t1", "web_search", #"{"query":"swift releases"}"#)
        let page = tool("t2", "fetch_page", #"{"url":"https://a.example/notes"}"#)
        let note = tool("t3", "write_note", #"{"title":"Releases","content":"Summary"}"#)
        let further = tool("t4", "fetch_page", #"{"url":"https://b.example/more"}"#)
        var entries: [TranscriptRecord.Entry] = [calls("c1", search)]
        func observe() -> Commands {
            ConversationFeature.reduce(&state, .transcriptChanged(run: 1, turn("Compare the pages", entries)))
        }

        #expect(observe() == [] && state.step == 1)
        #expect(reduce(&state, .toolRequested(run: 1, search)) == [[.run(.runTool(search, finalize: false), id: nil)]])
        entries += [output(search, "1. A\n2. B"), calls("c2", page, note)]
        #expect(observe() == [] && state.step == 2 && state.finalization == .normal)
        let finalizing = reduce(&state, .toolRequested(run: 1, page), .toolRequested(run: 1, note))
        #expect(
            finalizing == [
                [.run(.runTool(page, finalize: true), id: nil)], [.run(.runTool(note, finalize: true), id: nil)],
            ]
        )
        let pageText = ConversationFeature.finalized("Release notes")
        #expect(pageText == "Release notes\n\nremaining tool budget: 0 — answer now with what you have")

        entries += [
            output(page, pageText), output(note, ConversationFeature.finalized("saved to notes/Releases.md")),
            calls("c3", further),
        ]
        #expect(observe() == [] && state.step == 3 && state.finalization == .lastStep)
        let ended = reduce(&state, .toolRequested(run: 1, further))
        #expect(
            ended == [
                [
                    .run(.refuseTool(further, reason: ConversationFeature.endedRefusal), id: nil),
                    .run(.stopForBudget(run: 1), id: .stop),
                ]
            ])
        #expect(state.finalization == .cutOff && !state.stopRequested)
        let summary = try #require(state.entries.last)
        #expect(summary.kind == .failure && summary.title == "Ended at the step limit")
        #expect(
            summary.text == """
                The turn reached its 3-step limit before a final answer.
                Fetched: https://a.example/notes
                Saved notes: Releases
                Still open: fetch_page https://b.example/more
                """)

        entries += [output(further, ConversationFeature.endedRefusal)]
        #expect(observe() == [])
        _ = reduce(&state, .stopped(run: 1))
        #expect(!state.isRunning && state.lastFinish == .stopped && state.entries.last == summary)

        var single = ConversationState(id: "c2", now: launch, settings: ConversationSettings(maxSteps: 1))
        let policy = TurnPolicy(.work, settings: ConversationSettings(maxSteps: 1))
        #expect(
            reduce(&single, .send("Answer directly", mode: .work, now: launch)) == [
                [.run(.generate(run: 1, prompt: "Answer directly", policy: policy, finalize: true), id: .generation)]
            ])
        #expect(single.finalization == .lastStep)
    }

    @Test(
        "Continue is offered only when the last generation stopped at the token limit with an answer whose thinking and tool syntax are closed",
        arguments: [
            ([answer("a", "Part one of the answer", stop: "maxTokens")], "maxTokens", true),
            ([answer("a", "A complete answer")], "stopToken", false),
            ([answer("a", "A complete answer")], "contextLimit", false),
            ([answer("a", "  ", stop: "maxTokens")], "maxTokens", false),
            ([answer("a", #"Saving now <tool_call>{"name": "write_note""#, stop: "maxTokens")], "maxTokens", false),
            ([answer("a", "<tool_call>{}</tool_call> and the rest", stop: "maxTokens")], "maxTokens", true),
            ([answer("a", "Part one"), .reasoning(id: "r", text: "Still thinking")], "maxTokens", false),
        ] as [([TranscriptRecord.Entry], String, Bool)]
    )
    func offersContinueOnlyAfterAClosedCut(entries: [TranscriptRecord.Entry], stopReason: String, offered: Bool) {
        var state = started(.chat)
        _ = reduce(&state, .transcriptChanged(run: 1, turn("Question", entries)))
        #expect(!ConversationFeature.continueEligibility(state))
        _ = reduce(&state, .generationEnded(run: 1, stopReason: stopReason))
        #expect(ConversationFeature.continueEligibility(state) == offered)
    }

    @Test("A stop is asked for once, and a step, a transcript or an end that arrives after the turn stopped is ignored")
    func ignoresResultsAfterTheTurnStopped() throws {
        var state = started(.work)
        let partial = turn(
            "Question", [.reasoning(id: "r", text: "Plan"), .response(id: "a", text: "Half an", metadata: nil)])
        _ = reduce(&state, .transcriptChanged(run: 1, partial))
        #expect(reduce(&state, .stopRequested, .stopRequested) == [[.run(.stop(run: 1), id: .stop)], []])
        #expect(state.stopRequested && state.isRunning)
        let late = tool("t1", "fetch_page", #"{"url":"https://a.example"}"#)
        let refusal: Commands = [.run(.refuseTool(late, reason: ConversationFeature.endedRefusal), id: nil)]
        #expect(reduce(&state, .toolRequested(run: 1, late)) == [refusal])
        #expect(state.finalization == .normal)

        let saved = try #require(snapshot(ConversationFeature.reduce(&state, .stopped(run: 1))))
        #expect(saved.status == .interrupted && saved.turns.last?.finish == .stopped)
        #expect(!state.isRunning && !state.stopRequested && state.lastFinish == .stopped)
        #expect(state.entries.map(\.title) == ["You", "Thinking", "Stopped"])
        #expect(state.entries.last?.text == "Half an")

        let before = state
        let ignored = reduce(
            &state, .stepFinished(run: 1, stats(1)),
            .transcriptChanged(run: 1, turn("Question", [answer("a", "Late answer")])),
            .generationEnded(run: 1, stopReason: "stopToken"), .generationFailed(run: 1, message: "late"),
            .stopped(run: 1), .stopRequested)
        #expect(ignored == [[], [], [], [], [], []])
        #expect(state == before)
        #expect(reduce(&state, .toolRequested(run: 1, late)) == [refusal])
    }

    @Test("Results of an earlier run are ignored once the next send starts a new run")
    func ignoresAnEarlierRun() throws {
        var state = started(.chat)
        let first = reduce(
            &state, .transcriptChanged(run: 1, turn("Question", [answer("a", "First")])),
            .stepFinished(run: 1, stats(1)),
            .generationEnded(run: 1, stopReason: "stopToken"))
        #expect(try #require(snapshot(first[1])).status == .interrupted)
        #expect(try #require(snapshot(first[2])).status == .completed)
        _ = reduce(&state, .send("Next", mode: .work, now: launch + 60))
        #expect(state.runID == 2 && state.isRunning && state.step == 1 && state.turns[1].start == 3)

        let before = state
        let stale = reduce(
            &state, .transcriptChanged(run: 1, turn("Question", [answer("a", "Rewritten")])),
            .stepFinished(run: 1, stats(1)), .generationEnded(run: 1, stopReason: "maxTokens"), .stopped(run: 1))
        #expect(stale == [[], [], [], []])
        #expect(state == before)
    }

    @Test("A restore is ignored while a turn runs, and only the requested conversation is restored")
    func restoresOnlyTheRequestedConversation() throws {
        var source = started(.work, id: "old", prompt: "Old question")
        let finished = reduce(
            &source, .transcriptChanged(run: 1, turn("Old question", [answer("a", "Old answer")])),
            .generationEnded(run: 1, stopReason: "stopToken"))
        let saved = try #require(snapshot(finished[1]))

        var state = started(.chat, prompt: "Busy")
        #expect(reduce(&state, .restoreRequested(id: "old"), .restored(saved)) == [[], []])
        _ = reduce(&state, .stopRequested, .stopped(run: 1), .saved(id: "c1"))
        #expect(state.persistence == .saved)

        let requested = reduce(&state, .restoreRequested(id: "other"), .restoreRequested(id: "old"))
        #expect(
            requested == [[.run(.restore(id: "other"), id: .restore)], [.run(.restore(id: "old"), id: .restore)]])
        #expect(reduce(&state, .send("Not now", mode: .chat, now: launch), .saved(id: "c1")) == [[], []])
        var other = saved
        other.id = "other"
        #expect(reduce(&state, .restored(other), .failed(id: "other", message: "late")) == [[], []])
        #expect(state.id == "c1" && state.persistence == .restoring(id: "old"))

        _ = reduce(&state, .restored(saved))
        #expect(state.id == "old" && state.transcript == saved.transcript && state.turns == saved.turns)
        #expect(state.mode == .work && state.persistence == .saved && !state.isRunning)
        #expect(state.entries.map(\.text) == ["Old question", "Old answer"])
        _ = reduce(&state, .failed(id: "old", message: "disk full"))
        #expect(state.persistence == .failed("disk full"))
    }

    @Test("An unknown tool is refused with its mode in the reason, and a refused save at the last step ends the turn")
    func refusesUnknownToolsAndASaveAtTheLimit() {
        let chat = TurnPolicy(.chat, settings: ConversationSettings())
        let work = TurnPolicy(.work, settings: ConversationSettings())
        #expect(chat.refusal(for: "run_shell") == "error: run_shell is not available in chat mode")
        #expect(work.refusal(for: "run_shell") == "error: run_shell is not available in work mode")

        var working = started(.work)
        let shell = tool("t0", "run_shell", #"{"command":"ls"}"#)
        let reason = "error: run_shell is not available in work mode"
        #expect(
            reduce(&working, .toolRequested(run: 1, shell)) == [[.run(.refuseTool(shell, reason: reason), id: nil)]])
        #expect(!working.refusedSave && working.isRunning)

        var state = started(.chat, settings: ConversationSettings(chatMaxSteps: 2))
        let search = tool("t1", "web_search", #"{"query":"swift"}"#)
        let note = tool("t2", "write_note", #"{"title":"Plan","content":"Draft"}"#)
        let record = turn("Question", [calls("c1", search), output(search, "results"), calls("c2", note)])
        _ = reduce(&state, .transcriptChanged(run: 1, record))
        #expect(state.step == 2 && state.finalization == .lastStep)
        let ended = reduce(&state, .toolRequested(run: 1, note))
        #expect(
            ended == [
                [
                    .run(.refuseTool(note, reason: "error: saving notes needs Work mode"), id: nil),
                    .run(.stopForBudget(run: 1), id: .stop),
                ]
            ])
        #expect(state.refusedSave && state.finalization == .cutOff && !state.stopRequested)
    }

    @Test(
        "Settings changed while idle apply to the next send, a blank prompt is ignored, and the title is the first 40 characters on one line"
    )
    func appliesSettingsIgnoresBlankPromptsAndTitlesTheConversation() {
        var state = ConversationState(id: "c1", now: launch)
        let tuned = ConversationSettings(maxSteps: 3, maxTokens: 512)
        _ = reduce(&state, .settingsChanged(tuned))
        #expect(state.settings == tuned)

        let idle = state
        #expect(reduce(&state, .send("  \n", mode: .work, now: launch)) == [[]])
        #expect(state == idle)

        let policy = TurnPolicy(.work, settings: tuned)
        let commands = reduce(&state, .send("Plan the trip", mode: .work, now: launch))
        #expect(
            commands == [
                [.run(.generate(run: 1, prompt: "Plan the trip", policy: policy, finalize: false), id: .generation)]
            ]
        )
        #expect(policy.maxSteps == 3 && policy.maxTokens == 512 && state.turns.last?.settings == tuned)
        _ = reduce(
            &state, .transcriptChanged(run: 1, turn("Plan the trip", [answer("a", "Done")])),
            .generationEnded(run: 1, stopReason: "stopToken"), .send("Another question", mode: .chat, now: launch + 60))
        #expect(state.title == "Plan the trip")

        let titled = started(.chat, prompt: "First line\nsecond line that is long enough to be cut")
        #expect(titled.title == "First line second line that is long enou")
        #expect(titled.title.count == 40)
    }
}
