import Foundation

public enum TypedDecisionKind: String, Sendable, Hashable, Codable, CaseIterable {
    case noul
    case choice
    case score
}

public struct TypedDecisionRequest: Sendable {
    public var state: String
    public var question: String
    public var kind: TypedDecisionKind
    public var options: [String]

    public init(state: String, question: String, kind: TypedDecisionKind, options: [String]) {
        self.state = state
        self.question = question
        self.kind = kind
        self.options = options
    }
}

public enum TypedDecisionError: Error, Equatable, LocalizedError {
    case invalidBundle(String)
    case invalidOptions(String)
    case tooManyTokens(count: Int, limit: Int)
    case nonFinite(String)
    case invalidHidden(count: Int, expected: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidBundle(let message), .invalidOptions(let message), .nonFinite(let message): message
        case .tooManyTokens(let count, let limit):
            "The request renders to \(count) tokens; the model takes at most \(limit) tokens."
        case .invalidHidden(let count, let expected):
            "The head received \(count) values; it expects \(expected) values."
        }
    }
}

public struct TypedDecisionAnswer: Sendable, Equatable, Codable {
    public let kind: TypedDecisionKind
    public let options: [String]
    public let logits: [Float]
    public let probabilities: [Float]
    public let probabilitiesT1: [Float]
    public let top: Int
    public let tokenCount: Int

    public init(
        kind: TypedDecisionKind, options: [String], logits: [Float], probabilities: [Float], probabilitiesT1: [Float],
        top: Int, tokenCount: Int
    ) {
        self.kind = kind
        self.options = options
        self.logits = logits
        self.probabilities = probabilities
        self.probabilitiesT1 = probabilitiesT1
        self.top = top
        self.tokenCount = tokenCount
    }
}
