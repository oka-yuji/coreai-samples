public struct RecipeOptionDiagnostic: Sendable, Equatable, Codable, Identifiable {
    public var id: String { option }
    public let option: String
    public let probability: Float
    public let logit: Float

    public init(option: String, probability: Float, logit: Float) {
        self.option = option
        self.probability = probability
        self.logit = logit
    }
}

public struct RecipeQuestionDiagnostic: Sendable, Equatable, Codable, Identifiable {
    public let id: String
    public let instructions: String?
    public let selected: String
    public let options: [RecipeOptionDiagnostic]

    public init(id: String, instructions: String?, selected: String, options: [RecipeOptionDiagnostic]) {
        self.id = id
        self.instructions = instructions
        self.selected = selected
        self.options = options
    }
}

public struct RecipeDiagnostics: Sendable, Equatable, Codable {
    public let model: String
    public let inputTokens: Int
    public let questions: [RecipeQuestionDiagnostic]

    public init(model: String, inputTokens: Int, questions: [RecipeQuestionDiagnostic]) {
        self.model = model
        self.inputTokens = inputTokens
        self.questions = questions
    }
}
