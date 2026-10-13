import FeatureStore
import Foundation
import RecipeCore

public struct RecipeExample: Sendable, Equatable, Codable, Identifiable {
    public let id: String
    public let title: String
    public let photoPath: String
    public let recipeURL: String

    public init(id: String, title: String, photoPath: String, recipeURL: String) {
        self.id = id
        self.title = title
        self.photoPath = photoPath
        self.recipeURL = recipeURL
    }
}

public enum RecipePhase: Sendable, Equatable {
    case idle, analyzing, fetching, loading, checking, stopping, closed
}

public struct RecipeState: Sendable, Equatable {
    public var phase: RecipePhase = .idle
    public var session = 0
    public var revision = 0
    public var nextItemID = 0
    public var photo: FridgePhoto?
    public var pantry: [PantryItem] = []
    public var inventoryComplete = false
    public var kitchen: [KitchenItem]
    public var urlText = ""
    public var recipe: Recipe?
    public var modelName: String?
    public var modelPath: String
    public var report: RecipeReport?
    public var message: String?
    public var pendingRecipeURL: String?
    public var needsIdleBarrier = false
    public let vocabulary: FoodVocabulary
    public let examples: [RecipeExample]

    public init(vocabulary: FoodVocabulary, examples: [RecipeExample] = [], modelPath: String = "") {
        self.vocabulary = vocabulary
        self.examples = examples
        self.modelPath = modelPath
        kitchen = [
            "stove", "saucepan", "frying pan", "omelette pan", "knife", "cutting board", "bowl", "strainer",
            "heatproof dish", "oven", "toaster",
        ].map { KitchenItem(name: $0) }
    }

    public var busy: Bool { phase != .idle && phase != .closed }
    public var input: RecipeCheckInput? {
        recipe.map {
            RecipeCheckInput(recipe: $0, pantry: pantry, inventoryComplete: inventoryComplete, kitchen: kitchen)
        }
    }
    public var canCheck: Bool {
        phase == .idle && modelName != nil && input != nil && input?.problem == nil
            && (pantry.contains(where: \.confirmed) || inventoryComplete)
    }
}

public enum RecipeAction: Sendable, Equatable {
    case appeared
    case ended
    case importPhoto(URL)
    case selectExample(RecipeExample)
    case setURL(String)
    case fetchRecipe
    case openModel(String)
    case addItem
    case addRecipeIngredient(RecipeIngredient)
    case setName(Int, String)
    case setAmount(Int, String)
    case confirm(Int, Bool)
    case confirmListedItems
    case removeItem(Int)
    case setInventoryComplete(Bool)
    case setKitchen(String, KitchenStatus)
    case check
    case stop
    case photoLoaded(Int, FridgePhoto)
    case recipeLoaded(Int, Recipe)
    case modelLoaded(Int, String)
    case checked(Int, Int, RecipeReport)
    case failed(Int, String)
    case stopped(Int)
    case uiError(String)
}

public enum RecipeEffect: Sendable, Equatable {
    case inspect(Int, URL)
    case fetch(Int, String)
    case loadModel(Int, String)
    case check(Int, Int, RecipeCheckInput)
    case waitForIdle(Int)

    public var session: Int {
        switch self {
        case .inspect(let id, _), .fetch(let id, _), .loadModel(let id, _), .check(let id, _, _), .waitForIdle(let id):
            id
        }
    }
}

public enum RecipeEffectID: Sendable, Hashable {
    case operation, stopping
}

public enum RecipeFeature: Feature {
    public typealias State = RecipeState
    public typealias Action = RecipeAction
    public typealias Effect = RecipeEffect
    public typealias EffectID = RecipeEffectID
    public typealias Work = Command<Effect, EffectID>

    public static func reduce(_ state: inout State, _ action: Action) -> [Work] {
        if case .appeared = action {
            if state.phase == .closed {
                state.phase = .idle
                state.session += 1
                if state.needsIdleBarrier {
                    state.phase = .stopping
                    return [.run(.waitForIdle(state.session), id: .stopping)]
                }
            }
            if state.phase == .idle && state.modelName == nil && !state.modelPath.isEmpty {
                begin(&state, .loading)
                return [.run(.loadModel(state.session, state.modelPath), id: .operation)]
            }
            return []
        }
        guard state.phase != .closed else { return [] }
        switch action {
        case .appeared: return []
        case .ended:
            state.needsIdleBarrier = state.busy || state.needsIdleBarrier
            state.session += 1
            state.phase = .closed
            state.pendingRecipeURL = nil
            return [.cancelAll]
        case .stop:
            guard state.busy, state.phase != .stopping else { return [] }
            state.session += 1
            state.phase = .stopping
            state.pendingRecipeURL = nil
            return [.cancel(.operation), .run(.waitForIdle(state.session), id: .stopping)]
        case .photoLoaded(let session, let photo):
            guard session == state.session, state.phase == .analyzing else { return [] }
            state.photo = photo
            state.pantry = photo.ingredients.map { ingredient in
                defer { state.nextItemID += 1 }
                return PantryItem(
                    id: state.nextItemID, name: ingredient.name,
                    amount: state.vocabulary.suggestedAmount(for: ingredient))
            }
            state.inventoryComplete = false
            invalidate(&state)
            state.phase = .idle
            if let url = state.pendingRecipeURL {
                state.pendingRecipeURL = nil
                begin(&state, .fetching)
                return [.run(.fetch(state.session, url), id: .operation)]
            }
            return []
        case .recipeLoaded(let session, let recipe):
            guard session == state.session, state.phase == .fetching else { return [] }
            if state.recipe != recipe { expireRecipeAmounts(&state) }
            state.recipe = recipe
            state.urlText = recipe.source
            state.phase = .idle
            invalidate(&state)
            return []
        case .modelLoaded(let session, let name):
            guard session == state.session, state.phase == .loading else { return [] }
            state.modelName = name
            state.phase = .idle
            return []
        case .checked(let session, let revision, let report):
            guard session == state.session, revision == state.revision, state.phase == .checking else { return [] }
            state.report = report
            state.phase = .idle
            return []
        case .failed(let session, let message):
            guard session == state.session, state.busy, state.phase != .stopping else { return [] }
            state.message = message
            state.phase = .idle
            state.pendingRecipeURL = nil
            return []
        case .stopped(let session):
            guard session == state.session, state.phase == .stopping else { return [] }
            state.needsIdleBarrier = false
            state.phase = .idle
            state.message = "Stopped."
            return []
        default: break
        }
        guard !state.busy else { return [] }
        switch action {
        case .importPhoto(let url):
            state.pendingRecipeURL = nil
            invalidate(&state)
            begin(&state, .analyzing)
            return [.run(.inspect(state.session, url), id: .operation)]
        case .selectExample(let example):
            expireRecipeAmounts(&state)
            state.urlText = example.recipeURL
            state.recipe = nil
            state.pendingRecipeURL = example.recipeURL
            invalidate(&state)
            begin(&state, .analyzing)
            return [.run(.inspect(state.session, URL(filePath: example.photoPath)), id: .operation)]
        case .setURL(let text):
            guard text != state.urlText else { return [] }
            expireRecipeAmounts(&state)
            state.urlText = text
            state.recipe = nil
            invalidate(&state)
        case .fetchRecipe:
            begin(&state, .fetching)
            return [.run(.fetch(state.session, state.urlText), id: .operation)]
        case .openModel(let path):
            state.modelPath = path
            state.modelName = nil
            invalidate(&state)
            begin(&state, .loading)
            return [.run(.loadModel(state.session, path), id: .operation)]
        case .addItem:
            guard state.pantry.count < 32 else { return [] }
            state.pantry.append(PantryItem(id: state.nextItemID, name: ""))
            state.nextItemID += 1
            state.inventoryComplete = false
            invalidate(&state)
        case .addRecipeIngredient(let ingredient):
            guard state.pantry.count < 32 else { return [] }
            state.pantry.append(PantryItem(id: state.nextItemID, name: ingredient.name))
            state.nextItemID += 1
            state.inventoryComplete = false
            invalidate(&state)
        case .setName(let id, let value):
            guard let index = state.pantry.firstIndex(where: { $0.id == id }), state.pantry[index].name != value else {
                return []
            }
            state.pantry[index].name = value
            state.pantry[index].confirmed = false
            state.inventoryComplete = false
            invalidate(&state)
        case .setAmount(let id, let value):
            guard let index = state.pantry.firstIndex(where: { $0.id == id }), state.pantry[index].amount != value
            else { return [] }
            state.pantry[index].amount = value
            state.pantry[index].confirmed = false
            state.inventoryComplete = false
            invalidate(&state)
        case .confirm(let id, let value):
            guard let index = state.pantry.firstIndex(where: { $0.id == id }), state.pantry[index].confirmed != value
            else { return [] }
            state.pantry[index].confirmed = value
            if !value { state.inventoryComplete = false }
            invalidate(&state)
        case .confirmListedItems:
            guard state.pantry.contains(where: { !$0.confirmed }) else { return [] }
            for index in state.pantry.indices
            where !state.pantry[index].name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                state.pantry[index].confirmed = true
            }
            invalidate(&state)
        case .removeItem(let id):
            guard state.pantry.contains(where: { $0.id == id }) else { return [] }
            state.pantry.removeAll { $0.id == id }
            state.inventoryComplete = false
            invalidate(&state)
        case .setInventoryComplete(let complete):
            if complete && state.pantry.contains(where: { !$0.confirmed || $0.name.isEmpty }) {
                state.message = "Confirm or remove every inventory item first."
                return []
            }
            guard state.inventoryComplete != complete else { return [] }
            state.inventoryComplete = complete
            invalidate(&state)
        case .setKitchen(let name, let status):
            guard let index = state.kitchen.firstIndex(where: { $0.name == name }),
                state.kitchen[index].status != status
            else { return [] }
            state.kitchen[index].status = status
            invalidate(&state)
        case .check:
            guard state.canCheck, let input = state.input else { return [] }
            begin(&state, .checking)
            return [.run(.check(state.session, state.revision, input), id: .operation)]
        case .uiError(let message):
            state.message = message
        default: break
        }
        return []
    }

    private static func begin(_ state: inout State, _ phase: RecipePhase) {
        state.session += 1
        state.phase = phase
        state.message = nil
    }

    private static func expireRecipeAmounts(_ state: inout State) {
        for index in state.pantry.indices
        where state.pantry[index].confirmed
            && state.pantry[index].amount.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "enough"
        {
            state.pantry[index].confirmed = false
            state.inventoryComplete = false
        }
    }

    private static func invalidate(_ state: inout State) {
        state.revision += 1
        state.report = nil
        state.message = nil
    }
}
