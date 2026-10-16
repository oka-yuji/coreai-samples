import ConversationFeature
import ConversationRuntime
import Foundation
import FoundationModels
import Synchronization
import Testing

@Suite(.timeLimit(.minutes(1)))
struct ConversationTranscriptTests {
    @Test(
        "Budget finalization closes pending calls without replay and sends a tool-disabled request with the same history"
    )
    func finalAnswerWithoutReplay() async throws {
        let observed = ArchiveProbe()
        let tool = ArchiveTool(probe: observed)
        let original = Transcript(entries: [
            .instructions(
                Transcript.Instructions(
                    id: "i", segments: [.text(Transcript.TextSegment(id: "it", content: "Use the existing facts."))],
                    toolDefinitions: [Transcript.ToolDefinition(tool: tool)])),
            .prompt(
                Transcript.Prompt(id: "p", segments: [.text(Transcript.TextSegment(id: "pt", content: "Research"))])),
            .toolCalls(
                Transcript.ToolCalls(
                    id: "calls",
                    [
                        Transcript.ToolCall(
                            id: "pending", toolName: tool.name,
                            arguments: try GeneratedContent(json: #"{"text":"blue"}"#))
                    ])),
        ])
        let session = LanguageModelSession(model: ArchiveModel(probe: observed), tools: [tool], transcript: original)
        session.transcript = ConversationTranscript.closingPendingCalls(in: session.transcript)
        let closed = session.transcript
        #expect(closed.count == original.count + 1)
        #expect(closed.prefix(original.count).map(\.id) == original.map(\.id))
        #expect(ConversationTranscript.closingPendingCalls(in: closed).map(\.id) == closed.map(\.id))
        let response = try await session.respond(
            to: ConversationFeature.budgetAnswerPrompt,
            options: GenerationOptions(maximumResponseTokens: 128, toolCallingMode: .disallowed))
        #expect(response.content == "Continued")
        let values = observed.values.withLock { $0 }
        #expect(values.toolCalls == 0 && values.toolMode == .disallowed)
        #expect(values.transcript?.prefix(closed.count).map(\.id) == closed.map(\.id))
    }

    @Test(
        "A restored Foundation Models session receives old turns and never replays historical tool calls",
        arguments: [true, false])
    func continueWithoutReplay(hasOutput: Bool) async throws {
        let observed = ArchiveProbe()
        let tool = ArchiveTool(probe: observed)
        let record = TranscriptRecord(
            entries: [
                .instructions(id: "i", text: "Keep prior facts."),
                .prompt(id: "p", text: "Save the blue notebook"),
                .reasoning(id: "r", text: "I will save it."),
                .toolCalls(
                    id: "c", calls: [ToolCallRecord(id: "t", name: tool.name, arguments: #"{"text":"blue notebook"}"#)],
                    metadata: StepMetadata(stopReason: "stopToken", sharedTokens: 7)),
            ]
                + (hasOutput
                    ? [
                        .toolOutput(id: "t", name: tool.name, text: "Saved note 1"),
                        .response(id: "a", text: "Saved.", metadata: StepMetadata(stopReason: "stopToken")),
                    ] : []))
        let restored = try ConversationTranscript.restore(record, tools: [tool])
        let session = LanguageModelSession(model: ArchiveModel(probe: observed), tools: [tool], transcript: restored)
        let response = try await session.respond(to: "What color was it?")
        #expect(response.content == "Continued")
        let result = observed.values.withLock { $0 }
        #expect(result.toolCalls == 0)
        let prompt = try #require(result.transcript)
        #expect(prompt.count > restored.count)
        #expect(prompt.prefix(restored.count).map(\.id) == restored.map(\.id))
        let outputs = prompt.compactMap { entry -> Transcript.ToolOutput? in
            if case .toolOutput(let output) = entry { output } else { nil }
        }
        let output = try #require(outputs.first)
        guard case .text(let segment) = output.segments.first else {
            Issue.record("Missing restored tool result")
            return
        }
        #expect(segment.content == (hasOutput ? "Saved note 1" : ConversationTranscript.interruptedTool))
        guard case .instructions(let instructions) = restored.first else {
            Issue.record("Missing restored instructions")
            return
        }
        #expect(instructions.toolDefinitions.map(\.name) == [tool.name])
    }

    @Test("Multiple tool-call entries may include response text before their results arrive")
    func multipleCalls() throws {
        let record = TranscriptRecord(entries: [
            .toolCalls(id: "c1", calls: [ToolCallRecord(id: "a", name: "save_note", arguments: "{}")], metadata: nil),
            .response(id: "between", text: "Also save this.", metadata: nil),
            .toolCalls(id: "c2", calls: [ToolCallRecord(id: "b", name: "save_note", arguments: "{}")], metadata: nil),
            .toolOutput(id: "a", name: "save_note", text: "Saved A"),
            .toolOutput(id: "b", name: "save_note", text: "Saved B"),
        ])
        let restored = try ConversationTranscript.restore(record, tools: [])
        #expect(restored.map(\.id) == ["c1", "between", "c2", "a", "b"])
    }

    @Test("A malformed tool argument or unmatched output fails restoration instead of losing data")
    func rejectsBrokenTools() {
        let malformed = TranscriptRecord(entries: [
            .toolCalls(
                id: "c", calls: [ToolCallRecord(id: "t", name: "save_note", arguments: "{broken")], metadata: nil)
        ])
        #expect(throws: (any Error).self) { try ConversationTranscript.restore(malformed, tools: []) }
        let unmatched = TranscriptRecord(entries: [.toolOutput(id: "missing", name: "save_note", text: "Done")])
        #expect(throws: ConversationArchive.Failure.self) { try ConversationTranscript.restore(unmatched, tools: []) }
    }
}

private final class ArchiveProbe: Sendable {
    struct Values: Sendable {
        var toolCalls = 0
        var transcript: Transcript?
        var toolMode: GenerationOptions.ToolCallingMode.Kind?
    }
    let values = Mutex(Values())
}

private struct ArchiveTool: Tool {
    let name = "save_note"
    let description = "Save a note"
    let probe: ArchiveProbe

    @Generable
    struct Arguments {
        var text: String
    }

    func call(arguments: Arguments) async throws -> String {
        probe.values.withLock { $0.toolCalls += 1 }
        return "Unexpected replay"
    }
}

private struct ArchiveModel: LanguageModel {
    struct Executor: LanguageModelExecutor {
        typealias Model = ArchiveModel
        init(configuration: UUID) {}

        func respond(
            to request: LanguageModelExecutorGenerationRequest, model: Model,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            model.probe.values.withLock {
                $0.transcript = request.transcript
                $0.toolMode = request.generationOptions.toolCallingMode?.kind
            }
            await channel.send(.response(action: .appendText("Continued", tokenCount: 1)))
        }
    }

    let probe: ArchiveProbe
    let executorConfiguration = UUID()
    let capabilities = LanguageModelCapabilities([.toolCalling])
}
