import CoreAIBackend
import Foundation

struct CoreAIReceiptDecision: ReceiptDecisionExecuting {
    let model: TypedDecisionModel

    static func load(_ url: URL) async throws -> any ReceiptDecisionExecuting {
        CoreAIReceiptDecision(model: try await TypedDecisionModel.load(bundle: url))
    }

    func decide(_ input: ReceiptDecisionInput) async throws -> ReceiptDecisionValues {
        let answer = try await model.decide(
            TypedDecisionRequest(state: input.state, question: input.question, kind: .choice, options: input.options))
        return ReceiptDecisionValues(probabilities: answer.probabilities, tokenCount: answer.tokenCount)
    }
}
