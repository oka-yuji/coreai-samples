import FeatureStore
import Foundation
import RecipeCore
import RecipeFeature
import RecipeRuntime

@MainActor
enum RecipeAssembly {
    static let modelPathKey = "recipeDecisionBundlePath"

    static func makeStore() -> Store<RecipeFeature> {
        do {
            let state = try initialState()
            let runner = RecipeRunner(service: try LiveRecipeService()) {
                UserDefaults.standard.set($0, forKey: modelPathKey)
            }
            return Store(initialState: state, run: runner.run)
        } catch {
            let message = "The recipe demo could not open its resources: " + error.localizedDescription
            var state = RecipeState(vocabulary: FoodVocabulary(foods: [], units: [:]))
            state.message = message
            return Store(initialState: state) { effect, send in
                if case .waitForIdle(let session) = effect {
                    send(.stopped(session))
                } else {
                    send(.failed(effect.session, message))
                }
            }
        }
    }

    static func initialState() throws -> RecipeState {
        let path =
            ProcessInfo.processInfo.environment["COREAI_DECISION_BUNDLE"]
            ?? UserDefaults.standard.string(forKey: modelPathKey) ?? ""
        return try RecipeState(vocabulary: FoodVocabulary.bundled(), examples: examples(), modelPath: path)
    }

    static func examples() throws -> [RecipeExample] {
        guard let url = Bundle.main.url(forResource: "recipe-examples", withExtension: "json") else {
            throw RecipeError.invalid("The recipe examples are missing.")
        }
        let records = try JSONDecoder().decode([ExampleResource].self, from: Data(contentsOf: url))
        return try records.map { record in
            guard let photo = Bundle.main.url(forResource: record.id, withExtension: "png") else {
                throw RecipeError.invalid("The example photo is missing: " + record.id)
            }
            return RecipeExample(id: record.id, title: record.title, photoPath: photo.path, recipeURL: record.recipeURL)
        }
    }

    private struct ExampleResource: Decodable {
        let id: String
        let title: String
        let recipeURL: String
    }
}
