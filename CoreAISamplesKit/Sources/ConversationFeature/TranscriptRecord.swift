import Foundation

public struct TranscriptRecord: Codable, Equatable, Sendable {
    public enum Entry: Codable, Equatable, Sendable {
        case instructions(id: String, text: String)
        case prompt(id: String, text: String)
        case reasoning(id: String, text: String)
        case response(id: String, text: String, metadata: StepMetadata?)
        case toolCalls(id: String, calls: [ToolCallRecord], metadata: StepMetadata?)
        case toolOutput(id: String, name: String, text: String)

        public var id: String {
            switch self {
            case .instructions(let id, _), .prompt(let id, _), .reasoning(let id, _), .response(let id, _, _),
                .toolCalls(let id, _, _), .toolOutput(let id, _, _):
                id
            }
        }
    }

    public var entries: [Entry]

    public init(entries: [Entry] = []) {
        self.entries = entries
    }

    public func steps(from start: Int, to end: Int? = nil) -> [TranscriptStep] {
        let end = min(end ?? entries.count, entries.count)
        var steps: [TranscriptStep] = []
        var index = start
        while index < end {
            var outputs = index
            while outputs < end, !entries[outputs].isToolOutput { outputs += 1 }
            var after = outputs
            while after < end, entries[after].isToolOutput { after += 1 }
            steps.append(TranscriptStep(generated: index..<outputs, outputs: outputs < after ? outputs..<after : nil))
            index = after
        }
        return steps
    }
}

public struct TranscriptStep: Equatable, Sendable {
    public var generated: Range<Int>
    public var outputs: Range<Int>?

    public init(generated: Range<Int>, outputs: Range<Int>?) {
        self.generated = generated
        self.outputs = outputs
    }
}

public struct StepMetadata: Codable, Equatable, Sendable {
    public var stopReason: String?
    public var sharedTokens: Int?
    public var truncatedToolOutputs: Int?
    public var prefillSeconds: Double?
    public var decodeSeconds: Double?
    public var reservedTokens: Int?

    public init(
        stopReason: String? = nil, sharedTokens: Int? = nil, truncatedToolOutputs: Int? = nil,
        prefillSeconds: Double? = nil, decodeSeconds: Double? = nil, reservedTokens: Int? = nil
    ) {
        self.stopReason = stopReason
        self.sharedTokens = sharedTokens
        self.truncatedToolOutputs = truncatedToolOutputs
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
        self.reservedTokens = reservedTokens
    }
}

public struct ToolCallRecord: Codable, Equatable, Sendable {
    static let argumentOrder = [
        "web_search": ["query", "max_results"], "fetch_page": ["url", "query"],
        "write_note": ["title", "content"], "append_note": ["note_id", "part", "content"],
        "move_note": ["title", "destination"],
    ]

    public var id: String
    public var name: String
    public var arguments: String

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    var argumentLines: [(name: String, value: String)] {
        guard let object = (try? JSONSerialization.jsonObject(with: Data(arguments.utf8))) as? [String: Any] else {
            return arguments.isEmpty ? [] : [("arguments", arguments)]
        }
        let order = Self.argumentOrder[name] ?? []
        let keys = order.filter { object[$0] != nil } + object.keys.filter { !order.contains($0) }.sorted()
        return keys.compactMap { key in
            guard let value = object[key], !(value is NSNull) else { return nil }
            return (key, value as? String ?? Self.json(value))
        }
    }

    var summary: String {
        argumentLines.map { "\($0.name): \($0.value)" }.joined(separator: "\n")
    }

    func value(_ argument: String) -> String? {
        argumentLines.first { $0.name == argument }?.value
    }

    private static func json(_ value: Any) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed, .sortedKeys])
        else { return "\(value)" }
        return String(decoding: data, as: UTF8.self)
    }
}

extension TranscriptRecord.Entry {
    var isPrompt: Bool {
        if case .prompt = self { true } else { false }
    }

    var isReasoning: Bool {
        if case .reasoning = self { true } else { false }
    }

    var isToolOutput: Bool {
        if case .toolOutput = self { true } else { false }
    }

    var response: String? {
        if case .response(_, let text, _) = self { text } else { nil }
    }

    var calls: [ToolCallRecord] {
        if case .toolCalls(_, let calls, _) = self { calls } else { [] }
    }

    var metadata: StepMetadata? {
        switch self {
        case .response(_, _, let metadata), .toolCalls(_, _, let metadata): metadata
        default: nil
        }
    }
}
