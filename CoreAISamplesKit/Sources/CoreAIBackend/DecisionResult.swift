import Foundation

public struct DecisionProbability: Sendable, Equatable, Codable {
    public let option: String
    public let probability: Float
    public let logit: Float

    public init(option: String, probability: Float, logit: Float) {
        self.option = option
        self.probability = probability
        self.logit = logit
    }
}

public enum DecisionAnswer: Sendable, Equatable, Codable {
    case choice(id: String, distribution: [DecisionProbability], top: String)
    case noul(id: String, distribution: [DecisionProbability], probabilityTrue: Float)
    case score(id: String, distribution: [DecisionProbability], expectedValue: Float)

    public var id: String {
        switch self {
        case .choice(let id, _, _), .noul(let id, _, _), .score(let id, _, _): id
        }
    }

    public var distribution: [DecisionProbability] {
        switch self {
        case .choice(_, let distribution, _), .noul(_, let distribution, _), .score(_, let distribution, _):
            distribution
        }
    }

    public var top: String {
        if case .choice(_, _, let top) = self { return top }
        return distribution.max { $0.probability < $1.probability }?.option ?? ""
    }

    public var maximumProbability: Float {
        distribution.map(\.probability).max() ?? 0
    }
}

public struct DecisionResult: Sendable, Equatable, Codable {
    public let answers: [DecisionAnswer]
    public let model: String
    public let modelVersion: String
    public let inputTokens: Int
    public let elapsedSeconds: Double

    public init(
        answers: [DecisionAnswer], model: String, modelVersion: String, inputTokens: Int, elapsedSeconds: Double
    ) {
        self.answers = answers
        self.model = model
        self.modelVersion = modelVersion
        self.inputTokens = inputTokens
        self.elapsedSeconds = elapsedSeconds
    }

    static func answers(request: DecisionRequest, rendered: RenderedDecision, logits: [Float], stride: Int) throws
        -> [DecisionAnswer]
    {
        try request.questions.enumerated().map { index, question in
            let ids = rendered.optionIDs[index]
            let values = Array(logits[(index * stride)..<(index * stride + ids.count)])
            guard let maximum = values.max(), values.allSatisfy(\.isFinite) else {
                throw DecisionError.invalidOutput("The model returned non-finite logits.")
            }
            let weights = values.map { exp($0 - maximum) }
            let sum = weights.reduce(0, +)
            let distribution = ids.indices.map {
                DecisionProbability(option: ids[$0], probability: weights[$0] / sum, logit: values[$0])
            }
            switch question.criteria {
            case .choice(let options):
                let inputOrder = options.compactMap { option in
                    distribution.first { $0.option.utf8.elementsEqual(option.id.utf8) }
                }
                guard let best = inputOrder.max(by: { $0.probability < $1.probability }) else {
                    throw DecisionError.invalidOutput("The model returned no options.")
                }
                return .choice(id: question.id, distribution: distribution, top: best.option)
            case .noul:
                return .noul(id: question.id, distribution: distribution, probabilityTrue: distribution[0].probability)
            case .score:
                let expected = distribution.enumerated().reduce(Float(0)) {
                    $0 + Float($1.offset) * $1.element.probability
                }
                return .score(id: question.id, distribution: distribution, expectedValue: expected)
            }
        }
    }
}
