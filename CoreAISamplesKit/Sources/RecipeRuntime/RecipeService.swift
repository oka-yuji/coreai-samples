import CoreAIBackend
import Foundation
import RecipeCore

@MainActor
public protocol RecipeService {
    func inspect(_ url: URL) async throws -> FridgePhoto
    func fetch(_ url: String) async throws -> Recipe
    func loadModel(_ path: String) async throws -> String
    func check(_ input: RecipeCheckInput) async throws -> RecipeReport
    func waitForIdle() async
}

@MainActor
public final class LiveRecipeService: RecipeService {
    private let inspectPhoto: @Sendable (URL) async throws -> FridgePhoto
    private let getRecipe: @Sendable (String) async throws -> Recipe
    private let vocabulary: FoodVocabulary
    private let onQueued: (@MainActor () -> Void)?
    private var model: DecisionModel?
    private var context: DecisionContext?
    private var contextDocument: Data?
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init() throws {
        vocabulary = try FoodVocabulary.bundled()
        onQueued = nil
        let inspector = PhotoInspector()
        let fetcher = RecipeFetcher()
        inspectPhoto = { url in try await inspector.inspect(url) }
        getRecipe = { url in try await fetcher.fetch(url) }
    }

    init(
        vocabulary: FoodVocabulary,
        inspectPhoto: @escaping @Sendable (URL) async throws -> FridgePhoto,
        getRecipe: @escaping @Sendable (String) async throws -> Recipe,
        onQueued: (@MainActor () -> Void)? = nil
    ) {
        self.vocabulary = vocabulary
        self.inspectPhoto = inspectPhoto
        self.getRecipe = getRecipe
        self.onQueued = onQueued
    }

    public func inspect(_ url: URL) async throws -> FridgePhoto {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        let result = try await inspectPhoto(url)
        try Task.checkCancellation()
        return result
    }

    public func fetch(_ url: String) async throws -> Recipe {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        let result = try await getRecipe(url)
        try Task.checkCancellation()
        return result
    }

    public func loadModel(_ path: String) async throws -> String {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        model = nil
        context = nil
        contextDocument = nil
        let loaded = try await DecisionModel.load(bundle: URL(filePath: path))
        let name = await loaded.bundle.name
        try Task.checkCancellation()
        model = loaded
        return name
    }

    public func waitForIdle() async {
        await acquire()
        release()
    }

    public func check(_ input: RecipeCheckInput) async throws -> RecipeReport {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        if let problem = input.problem { throw RecipeError.invalid(problem) }
        guard let model else { throw RecipeError.invalid("Open a Clef-Flash decision bundle first.") }
        let started = ContinuousClock.now
        let questions = Self.questions(input)
        let document = input.recipe.document
        let documentKey = Data(document.utf8)
        var reused = context != nil && contextDocument == documentKey
        let result: DecisionResult
        do {
            if !reused {
                context = nil
                contextDocument = nil
                let created = try await model.context(state: document, questionBudget: 4_096)
                try Task.checkCancellation()
                context = created
                contextDocument = documentKey
            }
            guard let context else { throw RecipeError.invalid("The recipe context could not be created.") }
            result = try await context.decide(questions)
        } catch let error as DecisionError {
            switch error {
            case .questionBudgetExceeded, .tooManyTokens: break
            default: throw error
            }
            context = nil
            contextDocument = nil
            reused = false
            result = try await model.decide(DecisionRequest(state: document, questions: questions))
        }
        try Task.checkCancellation()
        guard result.answers.count == input.recipe.ingredients.count + 1,
            let equipmentAnswer = result.answers.last,
            let equipment = KitchenStatus(rawValue: equipmentAnswer.top)
        else { throw RecipeError.invalid("The decision model returned an incomplete recipe check.") }
        let ingredientAnswers = result.answers.dropLast()
        let requirements = try ingredientAnswers.map { answer in
            guard let requirement = RecipeRequirement(rawValue: answer.top) else {
                throw RecipeError.invalid("The decision model returned an unknown ingredient status.")
            }
            return requirement
        }
        let rows = try vocabulary.assessAll(
            input.recipe.ingredients, requirements: requirements,
            probabilities: ingredientAnswers.map(\.maximumProbability), pantry: input.pantry,
            complete: input.inventoryComplete)
        let duration = started.duration(to: .now).components
        return RecipeReport(
            rows: rows, equipment: equipment, equipmentProbability: equipmentAnswer.maximumProbability,
            seconds: Double(duration.seconds) + Double(duration.attoseconds) / 1e18,
            reusedTokens: reused ? result.reusedTokens : 0, modelVersion: result.modelVersion)
    }

    static func questions(_ input: RecipeCheckInput) -> [DecisionQuestion] {
        var questions = input.recipe.ingredients.map { ingredient in
            DecisionQuestion(
                id: "ingredient_\(ingredient.id)",
                instructions:
                    "Is \(ingredient.name) required by this recipe, or does the recipe permit omitting it? "
                    + "An optional garnish served only if desired can be omitted. "
                    + "Do not decide based on someone's inventory or invent changes to the recipe.",
                criteria: .choice([
                    DecisionOption(id: "required", description: "The ingredient is required to follow the recipe."),
                    DecisionOption(id: "uncertain", description: "The recipe does not make this clear."),
                    DecisionOption(id: "optional", description: "The recipe explicitly allows omitting it."),
                ]))
        }
        let kitchen = input.kitchen.map { $0.name + ": " + $0.status.rawValue }.joined(separator: ", ")
        questions.append(
            DecisionQuestion(
                id: "equipment",
                instructions:
                    "Can the recipe's stated method be followed with this kitchen? \(kitchen). "
                    + "Unlisted equipment is unknown. Respect alternatives: if an oven OR toaster is allowed, "
                    + "either available appliance is sufficient. Do not invent a different cooking method.",
                criteria: .choice([
                    DecisionOption(
                        id: "available", description: "All required equipment or stated alternatives are available."),
                    DecisionOption(
                        id: "unavailable",
                        description: "Required equipment and its stated alternatives are unavailable."),
                    DecisionOption(id: "unknown", description: "The required equipment has not been checked."),
                ])))
        return questions
    }

    private func acquire() async {
        if busy {
            await withCheckedContinuation {
                waiters.append($0)
                onQueued?()
            }
        } else {
            busy = true
        }
    }

    private func release() {
        if waiters.isEmpty { busy = false } else { waiters.removeFirst().resume() }
    }
}
