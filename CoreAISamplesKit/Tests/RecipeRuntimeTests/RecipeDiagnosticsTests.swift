import CoreAIBackend
import Foundation
import RecipeCore
import Testing

@testable import RecipeRuntime

@MainActor
struct RecipeDiagnosticsTests {
    @Test("Recipe inspection preserves the returned choice order, winner, probabilities and logits")
    func returnedScores() throws {
        let service = try LiveRecipeService()
        let input = input()
        let questions = LiveRecipeService.questions(input)
        let result = result()
        let report = try service.report(result, for: input, questions: questions, started: .now, reused: false)
        let diagnostics = try #require(report.diagnostics)
        #expect(diagnostics.model == result.model)
        #expect(diagnostics.inputTokens == 947)
        #expect(diagnostics.questions.map(\.id) == questions.map(\.id))
        #expect(diagnostics.questions.map(\.instructions) == questions.map(\.instructions))
        for (question, answer) in zip(diagnostics.questions, result.answers) {
            #expect(question.selected == answer.top)
            #expect(question.options.map(\.option) == answer.distribution.map(\.option))
            #expect(question.options.map(\.probability) == answer.distribution.map(\.probability))
            #expect(question.options.map(\.logit) == answer.distribution.map(\.logit))
        }
        #expect(report.rows.first?.amount == .short(2, "pcs"))
        #expect(report.rows.first?.requirement == .required)
        #expect(report.reusedTokens == 0)
        let reused = try service.report(result, for: input, questions: questions, started: .now, reused: true)
        #expect(reused.reusedTokens == 384)
        #expect(reused.rows == report.rows)
        #expect(reused.diagnostics == report.diagnostics)
        #expect(try JSONDecoder().decode(RecipeReport.self, from: JSONEncoder().encode(report)) == report)
    }

    @Test("Scores cannot be attached to a different question")
    func mismatchedQuestion() throws {
        let service = try LiveRecipeService()
        let input = input()
        let questions = LiveRecipeService.questions(input)
        #expect(throws: RecipeError.self) {
            try service.report(result(reversed: true), for: input, questions: questions, started: .now, reused: false)
        }
    }

    @Test("Reports saved before diagnostics were added still decode")
    func legacyReport() throws {
        let json = Data(
            #"{"rows":[],"equipment":"available","equipmentProbability":0.9,"seconds":1.5,"reusedTokens":0,"modelVersion":"v1"}"#
                .utf8)
        let report = try JSONDecoder().decode(RecipeReport.self, from: json)
        #expect(report.diagnostics == nil)
        #expect(report.seconds == 1.5)
        #expect(!report.needsReview)
    }

    @Test("Review uses a strict threshold and retains unknown-quantity checks")
    func reviewBoundary() {
        let threshold = RecipeReport.reviewThreshold
        #expect(makeReport(score: threshold.nextDown).needsReview)
        #expect(!makeReport(score: threshold).needsReview)
        #expect(makeReport(score: 1, equipmentScore: threshold.nextDown).needsReview)
        #expect(!makeReport(score: 1, equipmentScore: threshold).needsReview)
        #expect(makeReport(score: 1, amount: .unknown).needsReview)
        #expect(makeReport(score: 1, equipment: .unknown).needsReview)
    }

    private func makeReport(
        score: Float, equipmentScore: Float = 1, amount: AmountCheck = .sufficient,
        equipment: KitchenStatus = .available
    )
        -> RecipeReport
    {
        RecipeReport(
            rows: [
                RecipeCheckRow(
                    ingredient: RecipeIngredient(id: 0, name: "eggs", amount: "3 pcs"), decision: .have,
                    probability: score, amount: amount)
            ], equipment: equipment, equipmentProbability: equipmentScore, seconds: 0, reusedTokens: 0,
            modelVersion: "test")
    }

    private func input() -> RecipeCheckInput {
        RecipeCheckInput(
            recipe: Recipe(
                title: "Omelette", source: "", servings: "2", duration: "",
                ingredients: [RecipeIngredient(id: 0, name: "eggs", amount: "3 pcs")], steps: ["Cook the eggs."]),
            pantry: [PantryItem(id: 0, name: "eggs", amount: "1 pcs", confirmed: true)], inventoryComplete: true,
            kitchen: [KitchenItem(name: "frying pan", status: .available)])
    }

    private func result(reversed: Bool = false) -> DecisionResult {
        let answers: [DecisionAnswer] = [
            .choice(
                id: "ingredient_0",
                distribution: [
                    DecisionProbability(option: "optional", probability: 0.45, logit: -1.125),
                    DecisionProbability(option: "required", probability: 0.45, logit: -1.125),
                    DecisionProbability(option: "uncertain", probability: 0.1, logit: -2.629),
                ], top: "required"),
            .choice(
                id: "equipment",
                distribution: [
                    DecisionProbability(option: "available", probability: 0.85, logit: 2.5),
                    DecisionProbability(option: "unavailable", probability: 0.1, logit: 0.36),
                    DecisionProbability(option: "unknown", probability: 0.05, logit: -0.33),
                ], top: "available"),
        ]
        return DecisionResult(
            answers: reversed ? answers.reversed() : answers, model: "test-clef", modelVersion: "test-v1",
            inputTokens: 947, elapsedSeconds: 0.5, reusedTokens: 384)
    }
}
