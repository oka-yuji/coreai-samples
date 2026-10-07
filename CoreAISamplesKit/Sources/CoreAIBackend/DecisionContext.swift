import Foundation

public enum DecisionOverflowPolicy: Sendable {
    case reject
}

public struct DecisionContext: Sendable {
    private let model: DecisionModel
    private let generation: Int
    public let prefixTokenCount: Int
    public let questionBudget: Int
    public let key: String
    public let creationSeconds: Double

    init(
        model: DecisionModel, generation: Int, prefixTokenCount: Int, questionBudget: Int,
        key: String, creationSeconds: Double
    ) {
        self.model = model
        self.generation = generation
        self.prefixTokenCount = prefixTokenCount
        self.questionBudget = questionBudget
        self.key = key
        self.creationSeconds = creationSeconds
    }

    public func decide(_ questions: [DecisionQuestion]) async throws -> DecisionResult {
        try await model.decide(questions: questions, generation: generation)
    }
}

struct CachedDecisionDocument: Sendable {
    let generation: Int
    let state: String
    let prefix: [Int]
    let hidden: [Float16]
    let checkpoint: DecisionRuntime.Checkpoint
    let questionBudget: Int
    var previous: DecisionResult?
}
