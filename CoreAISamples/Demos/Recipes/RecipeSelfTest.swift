import FeatureStore
import Foundation
import RecipeCore
import RecipeFeature
import RecipeRuntime

@MainActor
enum RecipeSelfTest {
    static func run(directory: String) async {
        do {
            try await perform(directory: URL(filePath: directory))
        } catch {
            FileHandle.standardError.write(Data(("Recipe scenario failed: \(error.localizedDescription)\n").utf8))
            exit(1)
        }
    }

    private static func perform(directory: URL) async throws {
        let harness = try RecipeScenarioHarness()
        let store = harness.store
        defer { store.send(.ended) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        store.send(.appeared)
        await harness.settle()
        try require(store.state.modelName != nil, store.state.message ?? "Set COREAI_DECISION_BUNDLE.")
        var records: [Record] = []
        for example in store.state.examples {
            store.send(.selectExample(example))
            await harness.settle()
            guard let photo = store.state.photo, let recipe = store.state.recipe else {
                throw RecipeError.invalid(store.state.message ?? "Photo or recipe is missing.")
            }
            try require(
                !photo.ingredients.isEmpty && store.state.pantry.allSatisfy { !$0.confirmed },
                "Photo suggestions were not left for review.")
            for item in store.state.pantry { store.send(.removeItem(item.id)) }
            for name in inventory(example.id) {
                store.send(.addItem)
                guard let item = store.state.pantry.last else {
                    throw RecipeError.invalid("Inventory item was not added.")
                }
                store.send(.setName(item.id, name))
                store.send(.setAmount(item.id, example.id == "03-omelette" && name == "eggs" ? "1 pcs" : "enough"))
                store.send(.confirm(item.id, true))
            }
            store.send(.setInventoryComplete(example.id != "05-caprese"))
            for item in store.state.kitchen {
                store.send(.setKitchen(item.name, ["oven", "toaster"].contains(item.name) ? .unavailable : .available))
            }
            try require(store.state.canCheck, store.state.input?.problem ?? "Recipe is not ready.")
            store.send(.check)
            await harness.settle()
            guard let report = store.state.report else {
                throw RecipeError.invalid(store.state.message ?? "No recipe report.")
            }
            try verifyDiagnostics(report, recipe: recipe)
            records.append(
                Record(
                    id: example.id, source: recipe.source, suggestions: photo.ingredients, photoChecks: photo.checks,
                    photoSeconds: photo.seconds, scriptedReviewedInventory: store.state.pantry,
                    completeInventory: store.state.inventoryComplete, report: report))
            try write(records, to: directory.appendingPathComponent("five-store-cases.json"))
            switch example.id {
            case "01-oyakodon":
                try require(!report.hasShortage, "The reviewed ingredients should cover this recipe.")
            case "02-ginger-pork":
                try require(report.rows.last?.decision == .missing, "Ginger should be missing.")
            case "03-omelette":
                try require(report.rows.first?.amount == .short(2, "pcs"), "One egg should leave two eggs short.")
                try require(
                    photo.ingredients.contains { $0.name.lowercased().contains("egg") && $0.visibleCount == 1 },
                    "The photo should propose one egg.")
            case "04-gratin":
                try require(
                    report.equipment == .unavailable, "The missing oven and toaster should prevent the stated method.")
                store.send(.setKitchen("toaster", .available))
                try require(store.state.report == nil, "An equipment edit must invalidate the old result.")
                store.send(.check)
                await harness.settle()
                guard let updated = store.state.report else {
                    throw RecipeError.invalid("No toaster alternative result.")
                }
                try verifyDiagnostics(updated, recipe: recipe)
                try write(updated, to: directory.appendingPathComponent("toaster-store-case.json"))
                try require(
                    updated.equipment == .available && updated.reusedTokens > 0,
                    "The toaster alternative should reuse the recipe and satisfy the equipment requirement.")
            case "05-caprese":
                try require(report.rows[1].decision == .uncertain, "Unconfirmed mozzarella must remain unknown.")
                try require(
                    !photo.ingredients.contains { $0.name.lowercased().contains("cheese") },
                    "An opaque container must not become cheese.")
            default: throw RecipeError.invalid("Unexpected example.")
            }
        }
        try require(records.count == 5, "All five provided photos must be tested.")
        FileHandle.standardOutput.write(Data("Recipe Store/Runner/Service: all five scenarios passed.\n".utf8))
    }

    private static func inventory(_ id: String) -> [String] {
        switch id {
        case "01-oyakodon":
            ["chicken thigh", "onion", "eggs", "mitsuba", "cooked rice", "4x concentrated tsuyu", "water"]
        case "02-ginger-pork": ["pork", "onion", "soy sauce", "mirin", "vegetable oil"]
        case "03-omelette": ["eggs", "soy sauce", "sugar", "vegetable oil"]
        case "04-gratin":
            [
                "chicken thigh", "onion", "mushrooms", "broccoli", "macaroni", "salt", "vegetable oil", "pepper",
                "soy sauce", "white sauce", "milk", "shredded cheese", "breadcrumbs",
            ]
        default: ["tomato", "shiso", "soy sauce", "olive oil"]
        }
    }

    private static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw RecipeError.invalid(message) }
    }

    private static func verifyDiagnostics(_ report: RecipeReport, recipe: Recipe) throws {
        guard let diagnostics = report.diagnostics else { throw RecipeError.invalid("Missing diagnostics.") }
        let ids = recipe.ingredients.map { "ingredient_\($0.id)" } + ["equipment"]
        try require(diagnostics.inputTokens > 0 && diagnostics.questions.map(\.id) == ids, "Missing question metrics.")
        try require(!diagnostics.model.isEmpty, "Missing result model.")
        for question in diagnostics.questions {
            try require(question.options.count == 3, "Expected three returned choices.")
            try require(question.options.contains { $0.option == question.selected }, "Missing selected choice.")
            try require(
                question.options.allSatisfy { $0.probability.isFinite && $0.logit.isFinite }, "Invalid choice metrics.")
            try require(
                abs(question.options.reduce(Float(0)) { $0 + $1.probability } - 1) < 0.0001, "Invalid distribution.")
        }
    }

    private static func write(_ value: some Encodable, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(value).write(to: url, options: .atomic)
    }

    private struct Record: Encodable {
        let id: String
        let source: String
        let suggestions: [PhotoIngredient]
        let photoChecks: [String]
        let photoSeconds: Double
        let scriptedReviewedInventory: [PantryItem]
        let completeInventory: Bool
        let report: RecipeReport
    }
}

@MainActor
private final class RecipeScenarioHarness {
    let store: Store<RecipeFeature>
    private let completion: RecipeScenarioCompletion

    init() throws {
        let runner = RecipeRunner(service: try LiveRecipeService())
        let completion = RecipeScenarioCompletion()
        self.completion = completion
        store = try Store(initialState: RecipeAssembly.initialState()) { effect, send in
            await runner.run(effect, send: send)
            completion.signal()
        }
    }

    func settle() async {
        while store.state.busy { await completion.wait() }
    }
}

@MainActor
private final class RecipeScenarioCompletion {
    private var pending = 0
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func signal() {
        if waiters.isEmpty { pending += 1 } else { waiters.removeFirst().resume() }
    }

    func wait() async {
        if pending > 0 { pending -= 1 } else { await withCheckedContinuation { waiters.append($0) } }
    }
}
