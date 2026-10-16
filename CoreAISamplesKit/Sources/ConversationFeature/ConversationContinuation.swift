import Foundation

public struct IncompleteToolCall: Equatable, Sendable {
    public var name: String?

    public var isNote: Bool { name == "write_note" || name == "append_note" }

    public static func find(in text: String) -> Self? {
        let open = "<tool_call>"
        if let start = text.range(of: open, options: .backwards),
            !text[start.upperBound...].contains("</tool_call>")
        {
            let body = text[start.upperBound...]
            if let function = body.range(of: "<function="), let end = body[function.upperBound...].firstIndex(of: ">") {
                return Self(name: String(body[function.upperBound..<end]))
            }
            if let match = body.firstMatch(of: /"name"\s*:\s*"([^"]+)"/) { return Self(name: String(match.1)) }
            return Self(name: nil)
        }
        if [open, "</tool_call>"].contains(where: { tag in
            (2..<tag.count).contains(where: { text.hasSuffix(tag.prefix($0)) })
        }) {
            return Self(name: nil)
        }
        return nil
    }
}

extension ConversationFeature {
    public static let continuationPrompt = "Continue exactly where you stopped. Do not repeat."

    static func hasOpenThinking(_ text: String) -> Bool {
        let open = "<think>"
        let close = "</think>"
        if let start = text.range(of: open, options: .backwards), !text[start.upperBound...].contains(close) {
            return true
        }
        return [open, close].contains { tag in (2..<tag.count).contains { text.hasSuffix(tag.prefix($0)) } }
    }
}
