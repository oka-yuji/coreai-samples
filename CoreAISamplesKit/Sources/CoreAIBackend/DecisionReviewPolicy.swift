import Foundation

public struct DecisionReviewPolicy: Sendable, Decodable {
    public let threshold: Float
    public let unknownOptions: [String]

    public init(threshold: Float = 0.6, unknownOptions: [String] = []) {
        self.threshold = threshold
        self.unknownOptions = unknownOptions
    }

    public func requestsReview(_ answer: DecisionAnswer) -> Bool {
        answer.maximumProbability < threshold
            || unknownOptions.contains { $0.utf8.elementsEqual(answer.top.utf8) }
    }

    static func bundled() throws -> Self {
        guard let url = Bundle.module.url(forResource: "decision-review-policy", withExtension: "json") else {
            throw DecisionError.invalidBundle("The decision review policy is missing from the application.")
        }
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}

extension DecisionResult {
    static func changedQuestions(previous: DecisionResult?, answers: [DecisionAnswer], policy: DecisionReviewPolicy)
        -> [String]
    {
        guard let previous else { return [] }
        let old = Dictionary(uniqueKeysWithValues: previous.answers.map { (Array($0.id.utf8), $0) })
        let currentIDs = Set(answers.map { Array($0.id.utf8) })
        var changed = answers.filter { answer in
            guard let before = old[Array(answer.id.utf8)] else { return true }
            return !before.top.utf8.elementsEqual(answer.top.utf8)
                || policy.requestsReview(before) != policy.requestsReview(answer)
        }.map(\.id)
        changed += previous.answers.filter { !currentIDs.contains(Array($0.id.utf8)) }.map(\.id)
        return changed
    }
}
