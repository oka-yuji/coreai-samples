import ConversationFeature
import Foundation
import FoundationModels

public enum ConversationTranscript {
    public static let interruptedTool =
        "error: the previous turn was interrupted before this tool result was recorded. Do not repeat the operation automatically."

    public static func restore(_ record: TranscriptRecord, tools: [any Tool]) throws -> Transcript {
        var entries: [Transcript.Entry] = []
        var pending: [ToolCallRecord] = []
        func closePending() {
            entries += pending.map {
                .toolOutput(
                    Transcript.ToolOutput(
                        id: $0.id, toolName: $0.name, segments: segments(interruptedTool, id: $0.id)))
            }
            pending.removeAll()
        }
        for entry in record.entries {
            switch entry {
            case .prompt, .instructions: closePending()
            default: break
            }
            switch entry {
            case .instructions(let id, let text):
                entries.append(
                    .instructions(
                        Transcript.Instructions(
                            id: id, segments: segments(text, id: id),
                            toolDefinitions: tools.map { Transcript.ToolDefinition(tool: $0) })))
            case .prompt(let id, let text):
                entries.append(.prompt(Transcript.Prompt(id: id, segments: segments(text, id: id))))
            case .reasoning(let id, let text):
                entries.append(.reasoning(Transcript.Reasoning(id: id, segments: segments(text, id: id))))
            case .response(let id, let text, let values):
                entries.append(
                    .response(
                        Transcript.Response(
                            id: id, metadata: metadata(values), segments: segments(text, id: id))))
            case .toolCalls(let id, let calls, let values):
                let native = try calls.enumerated().map { index, call in
                    Transcript.ToolCall(
                        id: call.id, metadata: index == calls.count - 1 ? metadata(values) : [:],
                        toolName: call.name, arguments: try GeneratedContent(json: call.arguments))
                }
                entries.append(.toolCalls(Transcript.ToolCalls(id: id, native)))
                pending += calls
            case .toolOutput(let id, let name, let text):
                guard let index = pending.firstIndex(where: { $0.id == id && $0.name == name }) else {
                    throw ConversationArchive.Failure.invalidSnapshot
                }
                pending.remove(at: index)
                entries.append(
                    .toolOutput(Transcript.ToolOutput(id: id, toolName: name, segments: segments(text, id: id))))
            }
        }
        closePending()
        return Transcript(entries: entries)
    }

    private static func segments(_ text: String, id: String) -> [Transcript.Segment] {
        [.text(Transcript.TextSegment(id: id + "-text", content: text))]
    }

    private static func metadata(_ values: StepMetadata?) -> [String: any ConvertibleToGeneratedContent] {
        guard let values else { return [:] }
        var result: [String: any ConvertibleToGeneratedContent] = [:]
        result["stopReason"] = values.stopReason
        result["sharedTokens"] = values.sharedTokens
        result["truncatedToolOutputs"] = values.truncatedToolOutputs
        result["prefillSeconds"] = values.prefillSeconds
        result["decodeSeconds"] = values.decodeSeconds
        result["reservedTokens"] = values.reservedTokens
        return result
    }
}
