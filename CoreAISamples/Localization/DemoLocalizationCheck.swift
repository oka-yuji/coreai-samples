import DecideFeature
import Foundation
import RecipeCore
import RecipeFeature

@MainActor
enum DemoLocalizationCheck {
    static func run() {
        do {
            var keys = Demo.allCases.flatMap { [$0.title, $0.summary] }
            keys += AgentReasoningEffort.allCases.map(\.label)
            keys += DecideQuestionKind.allCases.map(\.title)
            keys += try RecipeAssembly.examples().map(\.title)
            guard let url = Bundle.main.url(forResource: "decide-examples", withExtension: "json") else {
                throw RecipeError.invalid("The decision examples are missing.")
            }
            keys += try JSONDecoder().decode([DecideExample].self, from: Data(contentsOf: url)).map(\.title)
            keys += KitchenStatus.allCases.map(\.title)
            keys += RecipeState(vocabulary: FoodVocabulary(foods: [], units: [:])).kitchen.map { $0.name.capitalized }
            keys += [IngredientDecision.have, .missing, .uncertain, .optional].map(\.title)
            keys += [RecipeRequirement.required, .optional, .uncertain].map(\.rawValue)
            keys += [ExtractRunner.notice, "Loaded", "Loading…", "Stopped.", "No match", "On hold"]
            keys += KitchenStatus.allCases.map(\.rawValue)
            keys += ["Kitchen equipment", "Open Clef-Flash bundle", "Change decision model"]
            keys += ["none", "Docs", "Thinking", "Thoughts", "Stopping…"]
            keys += ["Ended at the step limit", "Stopped", "Generation failed", "Where Work saves its notes"]
            keys += [
                "Could not save the conversation", "Read-only conversation", "Opening conversation…", "Save a copy",
                "Retry save", "Saving conversation…", "Keep app open",
            ]
            keys += Catalog.models.flatMap { [$0.memory, $0.platforms] }
            var failures: [String] = []
            for key in Set(keys).sorted() {
                let translated = Bundle.main.localizedString(
                    forKey: key, value: nil, table: nil, localizations: [Locale.Language(identifier: "ja")])
                if translated == key || translated.isEmpty { failures.append(key) }
            }
            for failure in failures { print("Missing runtime UI translation: \(failure)") }
            print("Runtime UI keys: \(Set(keys).count); missing Japanese translations: \(failures.count)")
            if !failures.isEmpty { exit(1) }
        } catch {
            FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
            exit(1)
        }
    }
}
