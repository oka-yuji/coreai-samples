import Foundation
import RecipeCore
import Testing

@testable import RecipeRuntime

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RECIPE_MODEL_TEST"] == "1"))
@MainActor
struct RecipeIntegrationTests {
    @Test("Five real photos and recipe URLs preserve the boundary between suggestions and reviewed inventory")
    func fiveCases() async throws {
        let environment = ProcessInfo.processInfo.environment
        let photos = try #require(environment["FRIDGE_PHOTOS_DIR"])
        let bundle = try #require(environment["COREAI_DECISION_BUNDLE"])
        let output = URL(filePath: environment["RECIPE_METRICS_DIR"] ?? "/private/tmp/fridge-logic-metrics")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let service = try LiveRecipeService()
        _ = try await service.loadModel(bundle)
        let cases = [
            (
                "01-oyakodon", "00006446",
                ["chicken thigh", "onion", "eggs", "mitsuba", "cooked rice", "4x concentrated tsuyu", "water"]
            ),
            ("02-ginger-pork", "00056818", ["pork", "onion", "soy sauce", "mirin", "vegetable oil"]),
            ("03-omelette", "00004002", ["eggs", "soy sauce", "sugar", "vegetable oil"]),
            (
                "04-gratin", "00004825",
                [
                    "chicken thigh", "onion", "mushrooms", "broccoli", "macaroni", "salt", "vegetable oil", "pepper",
                    "soy sauce", "white sauce", "milk", "shredded cheese", "breadcrumbs",
                ]
            ),
            ("05-caprese", "00001454", ["tomato", "shiso", "soy sauce", "olive oil"]),
        ]
        let equipment = [
            "stove", "saucepan", "frying pan", "omelette pan", "knife", "cutting board", "bowl", "strainer",
            "heatproof dish",
        ]
        var records: [[String: Any]] = []
        for (name, identifier, items) in cases {
            let photo = try await service.inspect(URL(filePath: photos).appendingPathComponent(name + ".png"))
            let recipe = try await service.fetch("https://www.kikkoman.co.jp/homecook/search/recipe/\(identifier)/")
            var pantry = items.enumerated().map {
                PantryItem(id: $0.offset, name: $0.element, amount: "enough", confirmed: true)
            }
            if name == "03-omelette" { pantry[0].amount = "1 pcs" }
            let kitchen =
                equipment.map { KitchenItem(name: $0, status: .available) }
                + [KitchenItem(name: "oven", status: .unavailable), KitchenItem(name: "toaster", status: .unavailable)]
            let input = RecipeCheckInput(
                recipe: recipe, pantry: pantry, inventoryComplete: name != "05-caprese", kitchen: kitchen)
            let report = try await service.check(input)
            let reportData = try JSONEncoder().encode(report)
            let photoData = try JSONEncoder().encode(photo.ingredients)
            let row: [String: Any] = [
                "case": name, "source": recipe.source,
                "photoSuggestions": try JSONSerialization.jsonObject(with: photoData),
                "photoChecks": photo.checks, "photoSeconds": photo.seconds,
                "reviewedInventoryIsScriptedTestInput": true,
                "report": try JSONSerialization.jsonObject(with: reportData),
            ]
            records.append(row)
            try JSONSerialization.data(withJSONObject: records, options: [.prettyPrinted, .sortedKeys])
                .write(to: output.appendingPathComponent("five-cases.json"), options: .atomic)
            #expect(!photo.ingredients.isEmpty)
            if name == "01-oyakodon" { #expect(!report.hasShortage) }
            if name == "02-ginger-pork" { #expect(report.rows.last?.decision == .missing) }
            if name == "03-omelette" {
                #expect(report.rows.first?.decision == .have)
                #expect(report.rows.first?.amount == .short(2, "pcs"))
                #expect(photo.ingredients.contains { $0.name.lowercased().contains("egg") && $0.visibleCount == 1 })
            }
            if name == "04-gratin" {
                #expect(report.equipment == .unavailable)
                let changed = kitchen.map {
                    KitchenItem(name: $0.name, status: $0.name == "toaster" ? .available : $0.status)
                }
                let reused = try await service.check(
                    RecipeCheckInput(recipe: recipe, pantry: pantry, inventoryComplete: true, kitchen: changed))
                #expect(reused.equipment == .available)
                try JSONEncoder().encode(reused).write(to: output.appendingPathComponent("gratin-toaster.json"))
            }
            if name == "05-caprese" {
                #expect(report.rows[1].decision == .uncertain)
                #expect(!photo.ingredients.contains { $0.name.lowercased().contains("cheese") })
            }
        }
    }
}
