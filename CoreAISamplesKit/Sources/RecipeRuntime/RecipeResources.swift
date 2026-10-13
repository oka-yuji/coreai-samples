import Foundation
import RecipeCore

extension FoodVocabulary {
    public static func bundled() throws -> Self {
        guard let url = Bundle.module.url(forResource: "recipe-vocabulary", withExtension: "json") else {
            throw RecipeError.invalid("The ingredient vocabulary is missing.")
        }
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}
