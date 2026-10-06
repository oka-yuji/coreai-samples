public struct DecisionRequest: Sendable, Hashable {
    public var state: String
    public var questions: [DecisionQuestion]

    public init(state: String, questions: [DecisionQuestion]) {
        self.state = state
        self.questions = questions
    }
}

public struct DecisionQuestion: Sendable, Hashable {
    public enum Criteria: Sendable, Hashable {
        case noul([String: String?] = [:])
        case choice([String: String?])
        case score([String?])
    }

    public var id: String
    public var instructions: String?
    public var criteria: Criteria

    public init(id: String, instructions: String? = nil, criteria: Criteria) {
        self.id = id
        self.instructions = instructions
        self.criteria = criteria
    }
}
