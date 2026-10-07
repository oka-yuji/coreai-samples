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
        case choice([DecisionOption])
        case score([String?])

        public static func choice(_ options: [String: String?]) -> Self {
            .choice(options.map { DecisionOption(id: $0.key, description: $0.value) })
        }
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

public struct DecisionOption: Sendable, Hashable {
    public var id: String
    public var description: String?

    public init(id: String, description: String? = nil) {
        self.id = id
        self.description = description
    }
}
