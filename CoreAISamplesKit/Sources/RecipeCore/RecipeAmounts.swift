import Foundation

public enum AmountCheck: Sendable, Equatable, Codable {
    case sufficient
    case short(Double, String)
    case unknown
    case notRequired

    public var isShort: Bool {
        if case .short = self { return true }
        return false
    }

    public var title: String {
        switch self {
        case .sufficient: "Amount checked"
        case .short(let amount, let unit):
            "Short by " + String(format: "%.2f", locale: Locale(identifier: "en_US_POSIX"), amount) + " " + unit
        case .unknown: "Amount not checked"
        case .notRequired: ""
        }
    }
}

public struct FoodVocabulary: Sendable, Equatable, Codable {
    public struct Food: Sendable, Equatable, Codable {
        public let id: String
        public let aliases: [String]
        public let countable: Bool
    }

    public let foods: [Food]
    public let units: [String: [String]]

    public init(foods: [Food], units: [String: [String]]) {
        self.foods = foods
        self.units = units
    }

    public func food(for name: String) -> Food? {
        let normalized = name.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        return foods.first { food in food.aliases.contains { normalized == $0.lowercased() } }
    }

    public func suggestedAmount(for ingredient: PhotoIngredient) -> String {
        guard let count = ingredient.visibleCount, count > 0, food(for: ingredient.name)?.countable == true else {
            return ""
        }
        return "\(count) pcs"
    }

    public func check(_ ingredient: RecipeIngredient, pantry: [PantryItem]) -> AmountCheck {
        let matches = matchingItems(for: ingredient, pantry: pantry)
        guard !matches.isEmpty else { return .unknown }
        if matches.contains(where: {
            $0.amount.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "enough"
        }) {
            return .sufficient
        }
        guard let required = quantity(ingredient.amount) else { return .unknown }
        let amounts = matches.compactMap { quantity($0.amount) }
        guard amounts.count == matches.count, amounts.allSatisfy({ $0.unit == required.unit }) else { return .unknown }
        let available = amounts.reduce(0) { $0 + $1.value }
        let missing = required.value - available
        return missing > 0.000_001 ? .short(missing, required.unit) : .sufficient
    }

    public func quantity(_ text: String) -> (value: Double, unit: String)? {
        let clean = text.replacingOccurrences(of: #"[\uFF08(].*$"#, with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let prefix = clean.range(of: #"^[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?"#, options: .regularExpression) else {
            return nil
        }
        let number = String(clean[prefix])
        let value: Double
        if number.contains("/") {
            let parts = number.split(separator: "/")
            guard parts.count == 2, let a = Double(parts[0]), let b = Double(parts[1]), b > 0 else { return nil }
            value = a / b
        } else {
            guard let parsed = Double(number) else { return nil }
            value = parsed
        }
        guard value.isFinite, value >= 0 else { return nil }
        let suffix = String(clean[prefix.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let unit = units.first(where: { $0.value.contains(suffix) })?.key else { return nil }
        return (value, unit)
    }

    public func matchingItems(for ingredient: RecipeIngredient, pantry: [PantryItem]) -> [PantryItem] {
        pantry.filter { item in
            guard item.confirmed else { return false }
            if let required = food(for: ingredient.name), let stocked = food(for: item.name) {
                return required.id == stocked.id
            }
            return item.name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                == ingredient.name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        }
    }

    public func assess(
        _ ingredient: RecipeIngredient, requirement: RecipeRequirement, probability: Float,
        pantry: [PantryItem], complete: Bool
    ) -> RecipeCheckRow {
        let decision: IngredientDecision
        let confirmed = pantry.filter(\.confirmed)
        let namesAreKnown =
            confirmed.isEmpty
            || (food(for: ingredient.name) != nil && confirmed.allSatisfy { food(for: $0.name) != nil })
        if requirement == .optional {
            decision = .optional
        } else if !matchingItems(for: ingredient, pantry: pantry).isEmpty {
            decision = .have
        } else if requirement == .required && complete && namesAreKnown {
            decision = .missing
        } else {
            decision = .uncertain
        }
        var amount = decision == .have ? check(ingredient, pantry: pantry) : .notRequired
        if requirement == .uncertain && amount.isShort { amount = .unknown }
        return RecipeCheckRow(
            ingredient: ingredient, decision: decision, probability: probability,
            amount: amount,
            requirement: requirement)
    }

    public func assessAll(
        _ ingredients: [RecipeIngredient], requirements: [RecipeRequirement], probabilities: [Float],
        pantry: [PantryItem], complete: Bool
    ) throws -> [RecipeCheckRow] {
        guard ingredients.count == requirements.count, ingredients.count == probabilities.count else {
            throw RecipeError.invalid("The recipe interpretation is incomplete.")
        }
        var rows = ingredients.indices.map { index in
            assess(
                ingredients[index], requirement: requirements[index], probability: probabilities[index], pantry: pantry,
                complete: complete)
        }
        let groups = Dictionary(
            grouping: ingredients.indices.filter {
                rows[$0].decision == .have && requirements[$0] == .required
            }
        ) { index in
            food(for: ingredients[index].name).map { "food:" + $0.id }
                ?? "name:" + ingredients[index].name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        }
        for indices in groups.values {
            guard let first = indices.first else { continue }
            let stock = matchingItems(for: ingredients[first], pantry: pantry)
            if stock.contains(where: {
                $0.amount.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "enough"
            }) {
                continue
            }
            let required = indices.compactMap { quantity(ingredients[$0].amount) }
            let available = stock.compactMap { quantity($0.amount) }
            let unit = required.first?.unit
            let comparable =
                required.count == indices.count && available.count == stock.count
                && required.allSatisfy { $0.unit == unit } && available.allSatisfy { $0.unit == unit }
            var remaining = available.reduce(0) { $0 + $1.value }
            for (offset, index) in indices.enumerated() {
                let amount: AmountCheck
                if comparable {
                    let needed = required[offset]
                    let shortage = max(0, needed.value - remaining)
                    remaining = max(0, remaining - needed.value)
                    amount = shortage > 0.000_001 ? .short(shortage, needed.unit) : .sufficient
                } else {
                    amount = .unknown
                }
                let old = rows[index]
                rows[index] = RecipeCheckRow(
                    ingredient: old.ingredient, decision: old.decision, probability: old.probability,
                    amount: amount, requirement: old.requirement)
            }
        }
        return rows
    }
}
