import Foundation

public enum DecideQuestionKind: String, Sendable, Codable, CaseIterable {
    case choice, noul, score

    public var title: String {
        switch self {
        case .choice: "Choice"
        case .noul: "Yes or no"
        case .score: "Score"
        }
    }
}

public struct DecideOption: Sendable, Equatable, Codable, Identifiable {
    public var id: Int
    public var key: String
    public var text: String

    public init(id: Int, key: String, text: String = "") {
        self.id = id
        self.key = key
        self.text = text
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id && lhs.key.utf8.elementsEqual(rhs.key.utf8) && lhs.text.utf8.elementsEqual(rhs.text.utf8)
    }
}

public struct DecideItem: Sendable, Equatable, Codable, Identifiable {
    public var id: Int
    public var key: String
    public var instructions: String
    public var kind: DecideQuestionKind
    public var options: [DecideOption]

    public init(id: Int, key: String, instructions: String = "", kind: DecideQuestionKind, options: [DecideOption]) {
        self.id = id
        self.key = key
        self.instructions = instructions
        self.kind = kind
        self.options = options
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id && lhs.key.utf8.elementsEqual(rhs.key.utf8)
            && lhs.instructions.utf8.elementsEqual(rhs.instructions.utf8) && lhs.kind == rhs.kind
            && lhs.options == rhs.options
    }

    public static func blank(id: Int) -> Self {
        Self(
            id: id, key: "Question \(id + 1)", kind: .choice,
            options: [
                DecideOption(id: 0, key: "yes", text: "Yes"),
                DecideOption(id: 1, key: "no", text: "No"),
                DecideOption(id: 2, key: "unknown", text: "Not stated in the document"),
            ])
    }
}

public struct DecideInput: Sendable, Equatable, Codable {
    public var document: String
    public var items: [DecideItem]

    public init(document: String, items: [DecideItem]) {
        self.document = document
        self.items = items
    }

    public var problem: String? {
        if document.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Add a document first." }
        if items.isEmpty || items.count > 16 { return "Use between 1 and 16 checklist items." }
        guard Set(items.map(\.id)).count == items.count, items.allSatisfy({ $0.id >= 0 && $0.id < Int.max }) else {
            return "Checklist item identifiers must be unique."
        }
        var seen: Set<[UInt8]> = []
        for item in items {
            if item.key.isEmpty || !seen.insert(Array(item.key.utf8)).inserted {
                return "Give each item a unique name."
            }
            if item.options.isEmpty || item.options.count > 8 { return "Use between 1 and 8 options per item." }
            if item.kind == .noul && item.options.count != 2 { return "Yes-or-no items need two options." }
            var keys: Set<[UInt8]> = []
            for option in item.options where item.kind == .choice {
                if option.key.isEmpty || !keys.insert(Array(option.key.utf8)).inserted {
                    return "Give each choice a unique, nonempty option ID."
                }
            }
        }
        return nil
    }
}

public struct DecideProbability: Sendable, Equatable, Codable, Identifiable {
    public let id: Int
    public let option: String
    public let label: String
    public let probability: Float

    public init(id: Int, option: String, label: String, probability: Float) {
        self.id = id
        self.option = option
        self.label = label
        self.probability = probability
    }
}

public struct DecideRow: Sendable, Equatable, Codable, Identifiable {
    public let id: Int
    public let key: String
    public let candidate: String
    public let probabilities: [DecideProbability]
    public let reviewFirst: Bool
    public let expectedScore: Float?

    public var maximumProbability: Float { probabilities.map(\.probability).max() ?? 0 }

    public init(
        id: Int, key: String, candidate: String, probabilities: [DecideProbability], reviewFirst: Bool,
        expectedScore: Float? = nil
    ) {
        self.id = id
        self.key = key
        self.candidate = candidate
        self.probabilities = probabilities
        self.reviewFirst = reviewFirst
        self.expectedScore = expectedScore
    }
}

public struct DecideOutput: Sendable, Equatable {
    public let rows: [DecideRow]
    public let modelName: String
    public let modelVersion: String
    public let inputTokens: Int
    public let reusedTokens: Int
    public let seconds: Double
    public let contextSeconds: Double
    public let reusedDocument: Bool
    public let recomputedAll: Bool
    public let requestJSON: String
    public let responseJSON: String

    public init(
        rows: [DecideRow], modelName: String, modelVersion: String, inputTokens: Int, reusedTokens: Int,
        seconds: Double, contextSeconds: Double = 0, reusedDocument: Bool, recomputedAll: Bool,
        requestJSON: String, responseJSON: String
    ) {
        self.rows = rows
        self.modelName = modelName
        self.modelVersion = modelVersion
        self.inputTokens = inputTokens
        self.reusedTokens = reusedTokens
        self.seconds = seconds
        self.contextSeconds = contextSeconds
        self.reusedDocument = reusedDocument
        self.recomputedAll = recomputedAll
        self.requestJSON = requestJSON
        self.responseJSON = responseJSON
    }
}

public struct DecideBundleInfo: Sendable, Equatable {
    public let name: String
    public let path: String
    public let tokenLimit: Int

    public init(name: String, path: String, tokenLimit: Int) {
        self.name = name
        self.path = path
        self.tokenLimit = tokenLimit
    }
}

public enum DecideExport {
    public enum ExportError: Error, LocalizedError {
        case invalidConfirmation

        public var errorDescription: String? { "A reviewer confirmation is not an allowed option." }
    }

    private struct Confirmation: Encodable {
        let question: String
        let option: String
        let status = "Confirmed"
    }

    public static func data(output: DecideOutput, confirmations: [Int: String]) throws -> Data {
        _ = try JSONSerialization.jsonObject(with: Data(output.requestJSON.utf8))
        _ = try JSONSerialization.jsonObject(with: Data(output.responseJSON.utf8))
        for (id, option) in confirmations {
            guard
                output.rows.first(where: { $0.id == id })?.probabilities.contains(
                    where: { $0.option.utf8.elementsEqual(option.utf8) }) == true
            else { throw ExportError.invalidConfirmation }
        }
        let confirmed = output.rows.compactMap { row in
            confirmations[row.id].map { Confirmation(question: row.key, option: $0) }
        }
        let human = String(decoding: try JSONEncoder().encode(confirmed), as: UTF8.self)
        return Data(
            ("{\"format\":\"decision-review/1\",\"request\":" + output.requestJSON
                + ",\"response\":" + output.responseJSON + ",\"reviewer_confirmations\":" + human + "}").utf8)
    }
}

public struct DecideExample: Sendable, Equatable, Codable, Identifiable {
    public let id: String
    public let title: String
    public let input: DecideInput
    public let extraItem: DecideItem
    public let searchHint: String

    public init(id: String, title: String, input: DecideInput, extraItem: DecideItem, searchHint: String) {
        self.id = id
        self.title = title
        self.input = input
        self.extraItem = extraItem
        self.searchHint = searchHint
    }
}

public struct DecideExportTicket: Sendable, Equatable, Hashable {
    public let id: Int
    public let session: Int
    public let revision: Int
    public let presentation: Int

    public init(id: Int, session: Int, revision: Int, presentation: Int) {
        self.id = id
        self.session = session
        self.revision = revision
        self.presentation = presentation
    }
}
