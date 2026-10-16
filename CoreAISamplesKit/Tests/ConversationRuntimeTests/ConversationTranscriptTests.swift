import ConversationFeature
import ConversationRuntime
import Foundation
import FoundationModels
import Synchronization
import Testing

@Suite(.timeLimit(.minutes(1)))
struct ConversationTranscriptTests {
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
            model.probe.values.withLock { $0.transcript = request.transcript }
            await channel.send(.response(action: .appendText("Continued", tokenCount: 1)))
        }
    }

    let probe: ArchiveProbe
    let executorConfiguration = UUID()
    let capabilities = LanguageModelCapabilities([.toolCalling])
}
