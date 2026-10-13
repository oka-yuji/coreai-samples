import Foundation
import RecipeCore
import Testing

@testable import RecipeRuntime

struct RecipeRuntimeTests {
    let source = "https://www.kikkoman.co.jp/homecook/search/recipe/00004002/"
    let page =
        #"<script type="application/ld+json">{"@type":"Recipe","name":"Test","recipeYield":"2","recipeIngredient":["egg 3pcs"],"recipeInstructions":["Cook in a pan."]}</script>"#

    @Test("A permitted recipe page is decoded from the actual transport result")
    func loadsPage() async throws {
        let data = Data(page.utf8)
        let fetcher = RecipeFetcher { _ in RecipeHTTPPage(status: 200, location: nil, data: data) }
        #expect(try await fetcher.fetch(source).title == "Test")
    }

    @Test("Every redirect destination is checked before the next network request")
    func rejectsRedirect() async {
        let fetcher = RecipeFetcher { _ in
            RecipeHTTPPage(status: 302, location: "http://127.0.0.1/private", data: Data())
        }
        await #expect(throws: RecipeError.self) { try await fetcher.fetch(source) }
    }

    @Test("Server failure, invalid UTF-8 and an oversized body are not replaced by sample data")
    func refusesFailedPages() async {
        for response in [
            RecipeHTTPPage(status: 503, location: nil, data: Data()),
            RecipeHTTPPage(status: 200, location: nil, data: Data([0xFF])),
            RecipeHTTPPage(status: 200, location: nil, data: Data(repeating: 32, count: 2_000_001)),
        ] {
            let fetcher = RecipeFetcher { _ in response }
            await #expect(throws: RecipeError.self) { try await fetcher.fetch(source) }
        }
    }

    @Test("Repeated redirects stop after the bounded number of requests")
    func redirectLimit() async {
        let fetcher = RecipeFetcher { url in RecipeHTTPPage(status: 302, location: url.absoluteString, data: Data()) }
        await #expect(throws: RecipeError.self) { try await fetcher.fetch(source) }
    }

    @Test("Ingredient identity and quantity use confirmed inventory instead of model guesses")
    @MainActor
    func confirmationBoundary() {
        let recipe = Recipe(
            title: "Test", source: source, servings: "2", duration: "",
            ingredients: [
                RecipeIngredient(id: 0, name: "cheese", amount: "30g")
            ], steps: ["Cook."])
        let input = RecipeCheckInput(
            recipe: recipe,
            pantry: [
                PantryItem(id: 0, name: "mystery cheese", confirmed: false),
                PantryItem(id: 1, name: "egg", confirmed: true),
            ],
            inventoryComplete: false, kitchen: [])
        let vocabulary = FoodVocabulary(foods: [], units: [:])
        let row = vocabulary.assess(
            recipe.ingredients[0], requirement: .required, probability: 0.99,
            pantry: input.pantry, complete: input.inventoryComplete)
        #expect(row.decision == .uncertain)
        let confirmed = vocabulary.assess(
            recipe.ingredients[0], requirement: .required, probability: 0.99,
            pantry: [PantryItem(id: 0, name: "cheese", amount: "enough", confirmed: true)], complete: false)
        #expect(confirmed.decision == .have)
        #expect(confirmed.amount == .sufficient)
    }

    @Test("Documented brands match their food, but a food flavored with soy sauce is not soy sauce")
    func exactAliases() throws {
        let url = try #require(Bundle.module.url(forResource: "alias-cases", withExtension: "json"))
        let cases = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: url))
        let vocabulary = try FoodVocabulary.bundled()
        #expect(vocabulary.food(for: try #require(cases["accepted"]))?.id == "soySauce")
        #expect(vocabulary.food(for: try #require(cases["plain"]))?.id == "soySauce")
        #expect(vocabulary.food(for: try #require(cases["differentFood"])) == nil)
    }
}
