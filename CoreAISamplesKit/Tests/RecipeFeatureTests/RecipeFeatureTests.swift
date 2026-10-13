import FeatureStore
import Foundation
import RecipeCore
import RecipeFeature
import Testing

struct RecipeFeatureTests {
    @Test("Photo suggestions require confirmation and a new photo clears complete-inventory consent")
    func photoReview() throws {
        var state = try makeState()
        state.inventoryComplete = true
        _ = RecipeFeature.reduce(&state, .importPhoto(URL(filePath: "/tmp/fridge.png")))
        let photo = FridgePhoto(
            image: Data(), filename: "fridge.png", ingredients: [PhotoIngredient(name: "eggs", visibleCount: 1)],
            checks: [], seconds: 1)
        _ = RecipeFeature.reduce(&state, .photoLoaded(state.session, photo))
        #expect(state.pantry.count == 1)
        #expect(state.pantry[0].amount == "1 pcs")
        #expect(!state.pantry[0].confirmed)
        #expect(!state.inventoryComplete)
        _ = RecipeFeature.reduce(&state, .setInventoryComplete(true))
        #expect(!state.inventoryComplete)
        _ = RecipeFeature.reduce(&state, .confirmListedItems)
        _ = RecipeFeature.reduce(&state, .setInventoryComplete(true))
        #expect(state.inventoryComplete)
        _ = RecipeFeature.reduce(&state, .setAmount(state.pantry[0].id, "2 pcs"))
        #expect(!state.pantry[0].confirmed)
        #expect(!state.inventoryComplete)
    }

    @Test("Every changed input invalidates a previous result", arguments: 0..<10)
    func invalidation(change: Int) throws {
        var state = try makeState()
        state.pantry = [PantryItem(id: 0, name: "eggs", amount: "1 pcs", confirmed: true)]
        state.nextItemID = 1
        state.report = report
        let actions: [RecipeAction] = [
            .setName(0, "egg"), .setAmount(0, "2 pcs"), .confirm(0, false), .removeItem(0), .addItem,
            .setInventoryComplete(true), .setKitchen("oven", .available), .setURL("https://example.com"),
            .openModel("/tmp/new-model"), .addRecipeIngredient(RecipeIngredient(id: 8, name: "salt", amount: "1 g")),
        ]
        _ = RecipeFeature.reduce(&state, actions[change])
        #expect(state.report == nil)
        #expect(state.revision == 1)
    }

    @Test("Stop waits for the idle barrier and ignores late successes and duplicate stop actions")
    func stopAndRetry() throws {
        var state = try makeState()
        state.inventoryComplete = true
        let check = RecipeFeature.reduce(&state, .check)
        #expect(check.count == 1)
        let canceledSession = state.session
        let commands = RecipeFeature.reduce(&state, .stop)
        #expect(commands == [.cancel(.operation), .run(.waitForIdle(state.session), id: .stopping)])
        #expect(state.phase == .stopping)
        #expect(RecipeFeature.reduce(&state, .stop).isEmpty)
        #expect(RecipeFeature.reduce(&state, .check).isEmpty)
        _ = RecipeFeature.reduce(&state, .checked(canceledSession, state.revision, report))
        #expect(state.report == nil)
        _ = RecipeFeature.reduce(&state, .stopped(canceledSession))
        #expect(state.phase == .stopping)
        _ = RecipeFeature.reduce(&state, .stopped(state.session))
        #expect(state.canCheck)
        #expect(!RecipeFeature.reduce(&state, .check).isEmpty)
        _ = RecipeFeature.reduce(&state, .failed(state.session, "Test failure"))
        #expect(state.canCheck)
        #expect(state.message == "Test failure")
    }

    @Test("Leaving cancels all effects and stale completions cannot mutate a reopened screen")
    func lifecycle() throws {
        var state = try makeState()
        _ = RecipeFeature.reduce(&state, .fetchRecipe)
        let oldSession = state.session
        #expect(RecipeFeature.reduce(&state, .ended) == [.cancelAll])
        #expect(state.phase == .closed)
        _ = RecipeFeature.reduce(&state, .appeared)
        let before = state
        _ = RecipeFeature.reduce(&state, .recipeLoaded(oldSession, recipe))
        _ = RecipeFeature.reduce(&state, .checked(oldSession, 0, report))
        _ = RecipeFeature.reduce(&state, .failed(oldSession, "Late failure"))
        #expect(state == before)
    }

    @Test("An example reads the actual photo before fetching its recipe, and a failure never installs a fallback")
    func exampleFlow() throws {
        var state = try makeState()
        let example = RecipeExample(id: "one", title: "Omelette", photoPath: "/tmp/egg.png", recipeURL: recipe.source)
        #expect(
            RecipeFeature.reduce(&state, .selectExample(example)) == [
                .run(.inspect(1, URL(filePath: example.photoPath)), id: .operation)
            ])
        #expect(state.recipe == nil)
        let photo = FridgePhoto(
            image: Data(), filename: "egg.png", ingredients: [], checks: ["Review the carton"], seconds: 1)
        #expect(
            RecipeFeature.reduce(&state, .photoLoaded(1, photo)) == [.run(.fetch(2, recipe.source), id: .operation)])
        _ = RecipeFeature.reduce(&state, .failed(2, "Offline"))
        #expect(state.phase == .idle)
        #expect(state.recipe == nil)
        #expect(state.pendingRecipeURL == nil)
        #expect(state.message == "Offline")
    }

    @Test("An unchanged field retains its confirmation, revision and result")
    func unchangedInput() throws {
        var state = try makeState()
        state.pantry = [PantryItem(id: 0, name: "eggs", amount: "1 pcs", confirmed: true)]
        state.report = report
        let before = state
        _ = RecipeFeature.reduce(&state, .setName(0, "eggs"))
        _ = RecipeFeature.reduce(&state, .setAmount(0, "1 pcs"))
        _ = RecipeFeature.reduce(&state, .confirm(0, true))
        #expect(state == before)
    }

    @Test("Recipe changes expire only recipe-relative amounts, including refreshes at the same URL")
    func recipeRelativeAmounts() throws {
        for changeURL in [false, true] {
            var state = try makeState()
            state.pantry = [
                PantryItem(id: 0, name: "eggs", amount: " ENOUGH ", confirmed: true),
                PantryItem(id: 1, name: "eggs", amount: "1 pcs", confirmed: true),
            ]
            state.inventoryComplete = true
            if changeURL { _ = RecipeFeature.reduce(&state, .setURL(recipe.source + "new")) }
            _ = RecipeFeature.reduce(&state, .fetchRecipe)
            let changed = Recipe(
                title: recipe.title, source: state.urlText, servings: "4", duration: "",
                ingredients: [RecipeIngredient(id: 0, name: "eggs", amount: "6 pcs")], steps: recipe.steps)
            _ = RecipeFeature.reduce(&state, .recipeLoaded(state.session, changed))
            #expect(!state.pantry[0].confirmed)
            #expect(state.pantry[0].amount == " ENOUGH ")
            #expect(state.pantry[1].confirmed)
            #expect(!state.inventoryComplete)
        }
        var state = try makeState()
        state.pantry = [PantryItem(id: 0, name: "eggs", amount: "enough", confirmed: true)]
        _ = RecipeFeature.reduce(&state, .fetchRecipe)
        _ = RecipeFeature.reduce(&state, .recipeLoaded(state.session, recipe))
        #expect(state.pantry[0].confirmed)
    }

    @Test("Reopening a screen with unfinished work preserves the idle barrier")
    func reopenedStopBarrier() throws {
        var state = try makeState()
        state.inventoryComplete = true
        _ = RecipeFeature.reduce(&state, .check)
        _ = RecipeFeature.reduce(&state, .stop)
        let canceledBarrier = state.session
        _ = RecipeFeature.reduce(&state, .ended)
        let commands = RecipeFeature.reduce(&state, .appeared)
        #expect(commands == [.run(.waitForIdle(state.session), id: .stopping)])
        #expect(state.phase == .stopping && !state.canCheck)
        _ = RecipeFeature.reduce(&state, .stopped(canceledBarrier))
        #expect(state.phase == .stopping)
        _ = RecipeFeature.reduce(&state, .stopped(state.session))
        #expect(state.canCheck && !state.needsIdleBarrier)
    }

    @Test("Changing a displayed ingredient translation keeps the source inventory and its confirmation")
    func translatedNameDoesNotEditInventory() throws {
        var state = try makeState()
        state.pantry = [PantryItem(id: 0, name: "eggs", amount: "1 pcs", confirmed: true)]
        state.inventoryComplete = true
        state.report = report
        let before = state
        _ = RecipeFeature.reduce(
            &state, .editDisplayedName(0, "localized eggs", previousDisplay: "localized eggs", originalSource: "eggs"))
        #expect(state == before)
        _ = RecipeFeature.reduce(&state, .editDisplayedName(0, "eggs", previousDisplay: "eggs", originalSource: "eggs"))
        #expect(state == before)
        _ = RecipeFeature.reduce(
            &state, .editDisplayedName(0, "onion", previousDisplay: "localized eggs", originalSource: "eggs"))
        #expect(state.pantry[0].name == "onion")
        #expect(!state.pantry[0].confirmed && !state.inventoryComplete)
        #expect(state.report == nil)
        #expect(state.revision == before.revision + 1)
    }

    @Test("Two edits through one displayed-name snapshot can return to the original ingredient")
    func consecutiveDisplayedEdits() throws {
        var state = try makeState()
        state.pantry = [PantryItem(id: 0, name: "eggs", amount: "1 pcs", confirmed: true)]
        state.inventoryComplete = true
        _ = RecipeFeature.reduce(
            &state, .editDisplayedName(0, "onion", previousDisplay: "localized eggs", originalSource: "eggs"))
        _ = RecipeFeature.reduce(
            &state, .editDisplayedName(0, "localized eggs", previousDisplay: "localized eggs", originalSource: "eggs"))
        #expect(state.pantry[0].name == "eggs")
        #expect(!state.pantry[0].confirmed && !state.inventoryComplete)
        #expect(state.revision == 2)
    }

    private func makeState() throws -> RecipeState {
        let vocabulary = try JSONDecoder().decode(
            FoodVocabulary.self,
            from: Data(#"{"foods":[{"id":"eggs","aliases":["eggs"],"countable":true}],"units":{"pcs":["pcs"]}}"#.utf8))
        var state = RecipeState(vocabulary: vocabulary)
        state.modelName = "test-model"
        state.recipe = recipe
        state.urlText = recipe.source
        return state
    }

    private var recipe: Recipe {
        Recipe(
            title: "Omelette", source: "https://www.kikkoman.co.jp/homecook/search/recipe/00004002/", servings: "2",
            duration: "", ingredients: [RecipeIngredient(id: 0, name: "eggs", amount: "3 pcs")],
            steps: ["Cook in a pan."])
    }

    private var report: RecipeReport {
        RecipeReport(
            rows: [], equipment: .available, equipmentProbability: 1, seconds: 1, reusedTokens: 0, modelVersion: "test")
    }
}
