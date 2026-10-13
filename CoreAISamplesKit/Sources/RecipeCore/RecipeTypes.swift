import Foundation

public enum RecipeError: Error, LocalizedError, Sendable, Equatable {
    case invalid(String)

    public var errorDescription: String? {
        switch self {
        case .invalid(let message): message
        }
    }
}

public struct RecipeIngredient: Sendable, Equatable, Codable, Identifiable {
    public let id: Int
    public let name: String
    public let amount: String

    public init(id: Int, name: String, amount: String) {
        self.id = id
        self.name = name
        self.amount = amount
    }

    public var line: String { amount.isEmpty ? name : name + " " + amount }
}

public struct Recipe: Sendable, Equatable, Codable {
    public let title: String
    public let source: String
    public let servings: String
    public let duration: String
    public let ingredients: [RecipeIngredient]
    public let steps: [String]

    public init(
        title: String, source: String, servings: String, duration: String,
        ingredients: [RecipeIngredient], steps: [String]
    ) {
        self.title = title
        self.source = source
        self.servings = servings
        self.duration = duration
        self.ingredients = ingredients
        self.steps = steps
    }

    public var document: String {
        ([title, "Servings: " + servings, "Ingredients:"] + ingredients.map(\.line)
            + ["Method:"] + steps).joined(separator: "\n")
    }
}

public struct PhotoIngredient: Sendable, Equatable, Codable {
    public let name: String
    public let visibleCount: Int?

    public init(name: String, visibleCount: Int?) {
        self.name = name
        self.visibleCount = visibleCount
    }
}

public struct FridgePhoto: Sendable, Equatable {
    public let image: Data
    public let filename: String
    public let ingredients: [PhotoIngredient]
    public let checks: [String]
    public let seconds: Double

    public init(image: Data, filename: String, ingredients: [PhotoIngredient], checks: [String], seconds: Double) {
        self.image = image
        self.filename = filename
        self.ingredients = ingredients
        self.checks = checks
        self.seconds = seconds
    }
}

public struct PantryItem: Sendable, Equatable, Codable, Identifiable {
    public let id: Int
    public var name: String
    public var amount: String
    public var confirmed: Bool

    public init(id: Int, name: String, amount: String = "", confirmed: Bool = false) {
        self.id = id
        self.name = name
        self.amount = amount
        self.confirmed = confirmed
    }
}

public enum KitchenStatus: String, Sendable, Equatable, Codable, CaseIterable {
    case available
    case unavailable
    case unknown

    public var title: String {
        switch self {
        case .available: "Have it"
        case .unavailable: "Do not have it"
        case .unknown: "Not checked"
        }
    }
}

public struct KitchenItem: Sendable, Equatable, Codable, Identifiable {
    public var id: String { name }
    public let name: String
    public var status: KitchenStatus

    public init(name: String, status: KitchenStatus = .unknown) {
        self.name = name
        self.status = status
    }
}

public struct RecipeCheckInput: Sendable, Equatable, Codable {
    public let recipe: Recipe
    public let pantry: [PantryItem]
    public let inventoryComplete: Bool
    public let kitchen: [KitchenItem]

    public init(recipe: Recipe, pantry: [PantryItem], inventoryComplete: Bool, kitchen: [KitchenItem]) {
        self.recipe = recipe
        self.pantry = pantry
        self.inventoryComplete = inventoryComplete
        self.kitchen = kitchen
    }

    public var problem: String? {
        if pantry.count > 32 { return "Keep at most 32 inventory items." }
        if pantry.contains(where: { $0.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) {
            return "Give each inventory item a name, or remove the empty item."
        }
        if pantry.contains(where: { $0.name.count > 120 || $0.amount.count > 60 }) {
            return "Use short ingredient names and amounts."
        }
        if inventoryComplete && pantry.contains(where: { !$0.confirmed }) {
            return "Confirm every inventory item before marking the inventory complete."
        }
        return nil
    }
}

public enum IngredientDecision: String, Sendable, Equatable, Codable {
    case have
    case missing
    case uncertain
    case optional

    public var title: String {
        switch self {
        case .have: "Have ingredient"
        case .missing: "Missing ingredient"
        case .uncertain: "Check ingredient"
        case .optional: "Optional"
        }
    }
}

public enum RecipeRequirement: String, Sendable, Equatable, Codable {
    case required
    case optional
    case uncertain
}

public struct RecipeCheckRow: Sendable, Equatable, Codable, Identifiable {
    public let id: Int
    public let ingredient: RecipeIngredient
    public let decision: IngredientDecision
    public let probability: Float
    public let amount: AmountCheck
    public let requirement: RecipeRequirement

    public init(
        ingredient: RecipeIngredient, decision: IngredientDecision, probability: Float, amount: AmountCheck,
        requirement: RecipeRequirement = .required
    ) {
        id = ingredient.id
        self.ingredient = ingredient
        self.decision = decision
        self.probability = probability
        self.amount = amount
        self.requirement = requirement
    }
}

public struct RecipeReport: Sendable, Equatable, Codable {
    public static let reviewThreshold: Float = 0.6

    public let rows: [RecipeCheckRow]
    public let equipment: KitchenStatus
    public let equipmentProbability: Float
    public let seconds: Double
    public let reusedTokens: Int
    public let modelVersion: String
    public let diagnostics: RecipeDiagnostics?

    public init(
        rows: [RecipeCheckRow], equipment: KitchenStatus, equipmentProbability: Float,
        seconds: Double, reusedTokens: Int, modelVersion: String, diagnostics: RecipeDiagnostics? = nil
    ) {
        self.rows = rows
        self.equipment = equipment
        self.equipmentProbability = equipmentProbability
        self.seconds = seconds
        self.reusedTokens = reusedTokens
        self.modelVersion = modelVersion
        self.diagnostics = diagnostics
    }

    public var hasShortage: Bool {
        rows.contains { $0.decision == .missing || $0.amount.isShort }
    }

    public var needsReview: Bool {
        equipment == .unknown || equipmentProbability < Self.reviewThreshold
            || rows.contains {
                $0.decision == .uncertain || $0.requirement == .uncertain || $0.probability < Self.reviewThreshold
                    || ($0.decision == .have && $0.amount == .unknown)
            }
    }

    public var title: String {
        if hasShortage { return "Something is missing" }
        if equipment == .unavailable { return "Check your kitchen equipment" }
        if needsReview { return "A few things still need checking" }
        return "Your confirmed inventory fits this recipe"
    }
}
