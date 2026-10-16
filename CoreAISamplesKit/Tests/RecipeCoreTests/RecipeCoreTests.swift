import Foundation
import Testing

@testable import RecipeCore

struct RecipeCoreTests {
    let source = "https://www.kikkoman.co.jp/homecook/search/recipe/00004002/"

    func vocabulary() -> FoodVocabulary {
        FoodVocabulary(
            foods: [
                .init(id: "egg", aliases: ["egg", "eggs"], countable: true),
                .init(id: "onion", aliases: ["onion"], countable: true),
                .init(id: "milk", aliases: ["milk"], countable: false),
                .init(id: "cheese", aliases: ["cheese"], countable: false),
            ], units: ["pcs": ["pcs", "pc"], "g": ["g"], "ml": ["ml"]])
    }

    func html(ingredients: [String] = ["egg 3 pcs", "soy sauce 5 ml"], title: String = "Test recipe") throws -> String {
        let value: [String: Any] = [
            "@graph": [
                [
                    "@type": ["Recipe"], "name": title, "recipeYield": "2 servings", "totalTime": "PT10M",
                    "recipeIngredient": ingredients,
                    "recipeInstructions": [
                        [
                            "@type": "HowToSection",
                            "itemListElement": [
                                ["@type": "HowToStep", "text": "Mix &amp; cook in a pan."]
                            ],
                        ]
                    ],
                ]
            ]
        ]
        let json = try JSONSerialization.data(withJSONObject: value)
        let body = try #require(String(data: json, encoding: .utf8))
        return "<script data-example='yes' type='application/ld+json'>" + body + "</script>"
    }

    @Test("A recipe preserves its source, servings, ingredients and nested method")
    func parsesRecipe() throws {
        let recipe = try RecipeParser.parse(html: html(), source: source)
        #expect(recipe.title == "Test recipe")
        #expect(recipe.source == source)
        #expect(recipe.servings == "2 servings")
        #expect(recipe.ingredients[0] == RecipeIngredient(id: 0, name: "egg", amount: "3 pcs"))
        #expect(recipe.ingredients[1] == RecipeIngredient(id: 1, name: "soy sauce", amount: "5 ml"))
        #expect(recipe.steps == ["Mix & cook in a pan."])
    }

    @Test("Invalid recipe pages never become empty successful recipes")
    func rejectsIncompletePages() throws {
        let pages = [
            "<html>no recipe</html>", "<script type='application/ld+json'>{bad</script>",
            try html(ingredients: []), try html(title: "  "), try html(ingredients: [" "]),
            try html(ingredients: Array(repeating: "egg 1pc", count: 15)),
        ]
        for page in pages {
            #expect(throws: RecipeError.self) { try RecipeParser.parse(html: page, source: source) }
        }
    }

    @Test("Unsupported schemes, hosts, paths, credentials and ports are rejected before fetching")
    func rejectsUnsupportedURLs() throws {
        for bad in [
            "http://www.kikkoman.co.jp/homecook/search/recipe/00004002/",
            "https://www.kikkoman.co.jp.evil.test/homecook/search/recipe/00004002/",
            "https://user:password@www.kikkoman.co.jp/homecook/search/recipe/00004002/",
            "https://www.kikkoman.co.jp:8080/homecook/search/recipe/00004002/",
            source + "?redirect=http://127.0.0.1/", "file:///private/tmp/recipe.html",
            "https://127.0.0.1/homecook/search/recipe/00004002/",
            "https://www.kikkoman.co.jp/homecook/search/recipe/%30%30%30%30%34%30%30%32/",
        ] { #expect(throws: RecipeError.self) { try RecipeURL.validate(bad) } }
        #expect(try RecipeURL.validate(source).host == "www.kikkoman.co.jp")
    }

    @Test("One confirmed egg is two short of the three eggs required for the unchanged recipe")
    func countsShortage() throws {
        let vocabulary = vocabulary()
        let ingredient = RecipeIngredient(id: 0, name: "egg", amount: "3pcs")
        let pantry = [PantryItem(id: 0, name: "Eggs", amount: "1 pc", confirmed: true)]
        #expect(vocabulary.check(ingredient, pantry: pantry) == .short(2, "pcs"))
        #expect(vocabulary.check(ingredient, pantry: [PantryItem(id: 0, name: "Eggs", amount: "1 pc")]) == .unknown)
    }

    @Test("Fractions use only matching units and unknown amounts never become sufficient")
    func comparesAmountsConservatively() throws {
        let vocabulary = vocabulary()
        let ingredient = RecipeIngredient(id: 0, name: "onion", amount: "1/4pcs")
        #expect(
            vocabulary.check(ingredient, pantry: [PantryItem(id: 0, name: "onion", amount: "1pcs", confirmed: true)])
                == .sufficient)
        for amount in ["", "1/0 pcs", "many", "100g", "-1pcs"] {
            #expect(
                vocabulary.check(
                    ingredient, pantry: [PantryItem(id: 0, name: "onion", amount: amount, confirmed: true)]) == .unknown
            )
        }
        #expect(vocabulary.suggestedAmount(for: PhotoIngredient(name: "milk", visibleCount: 1)).isEmpty)
        #expect(vocabulary.suggestedAmount(for: PhotoIngredient(name: "eggs", visibleCount: 3)) == "3 pcs")
    }

    @Test("An explicit sufficient amount belongs to the confirmed ingredient only")
    func confirmedEnough() throws {
        let vocabulary = vocabulary()
        let ingredient = RecipeIngredient(id: 0, name: "milk", amount: "150ml")
        #expect(
            vocabulary.check(ingredient, pantry: [PantryItem(id: 0, name: "milk", amount: "enough", confirmed: true)])
                == .sufficient)
        #expect(
            vocabulary.check(ingredient, pantry: [PantryItem(id: 0, name: "cheese", amount: "enough", confirmed: true)])
                == .unknown)
        #expect(vocabulary.food(for: "raw rice") == nil)
    }

    @Test("An optional ingredient never makes the recipe short when a smaller amount is added")
    func optionalAmount() {
        let ingredient = RecipeIngredient(id: 0, name: "onion", amount: "4 pcs")
        let row = vocabulary().assess(
            ingredient, requirement: .optional, probability: 0.99,
            pantry: [PantryItem(id: 0, name: "onion", amount: "1 pcs", confirmed: true)], complete: true)
        #expect(row.decision == .optional)
        #expect(row.amount == .notRequired)
    }

    @Test("An ingredient repeated in the recipe cannot spend the same stock twice")
    func repeatedIngredient() throws {
        let rows = try vocabulary().assessAll(
            [
                RecipeIngredient(id: 0, name: "egg", amount: "2 pcs"),
                RecipeIngredient(id: 1, name: "eggs", amount: "2 pcs"),
            ], requirements: [.required, .required], probabilities: [1, 1],
            pantry: [PantryItem(id: 0, name: "eggs", amount: "2 pcs", confirmed: true)], complete: true)
        #expect(rows.map(\.amount) == [.sufficient, .short(2, "pcs")])
    }

    @Test("An uncertain ingredient requirement cannot consume stock and create a definite shortage")
    func uncertainDemand() throws {
        let rows = try vocabulary().assessAll(
            [
                RecipeIngredient(id: 0, name: "egg", amount: "2 pcs"),
                RecipeIngredient(id: 1, name: "eggs", amount: "2 pcs"),
            ], requirements: [.uncertain, .required], probabilities: [1, 1],
            pantry: [PantryItem(id: 0, name: "eggs", amount: "2 pcs", confirmed: true)], complete: true)
        let report = RecipeReport(
            rows: rows, equipment: .available, equipmentProbability: 1,
            seconds: 0, reusedTokens: 0, modelVersion: "test")
        #expect(!report.hasShortage)
        #expect(report.needsReview)
        #expect(rows[1].amount == .sufficient)
    }

    @Test("Unrecognized names remain uncertain instead of proving an ingredient absent")
    func unrecognizedName() {
        let row = vocabulary().assess(
            RecipeIngredient(id: 0, name: "onion", amount: "1 pcs"),
            requirement: .required, probability: 1,
            pantry: [PantryItem(id: 0, name: "yellow onion", amount: "1 pcs", confirmed: true)], complete: true)
        #expect(row.decision == .uncertain)
    }

    @Test("Section text and child steps are both preserved, and unreadable steps refuse the whole recipe")
    func completeMethod() throws {
        let base = #"{"@type":"Recipe","name":"Test","recipeIngredient":["egg 1pcs"],"recipeInstructions":STEPS}"#
        func wrap(_ steps: String) -> String {
            "<script type='application/ld+json'>" + base.replacingOccurrences(of: "STEPS", with: steps) + "</script>"
        }
        let full = wrap(
            #"[{"@type":"HowToSection","text":"Prepare the food.","itemListElement":[{"text":"Bake in an oven."}]}]"#)
        #expect(try RecipeParser.parse(html: full, source: source).steps == ["Prepare the food.", "Bake in an oven."])
        for bad in [#"[{"text":"Prepare."},{"name":"Bake in an oven."}]"#, #"[{"text":"Prepare."},{"text":" "}]"#] {
            #expect(throws: RecipeError.self) { try RecipeParser.parse(html: wrap(bad), source: source) }
        }
    }
}
