import FeatureStore
import Foundation
import RecipeCore
import RecipeFeature
import RecipeRuntime

@MainActor
enum RecipeAssembly {
    static let modelPathKey = "recipeDecisionBundlePath"

    static func makeStore(managed: Bool = false) -> Store<RecipeFeature> {
        do {
            let state = try initialState(managed: managed)
            let service = try LiveRecipeService(acquireModel: { path in
                guard managed else { return {} }
                return try await MacModelLibraryServices.acquire(path, modelID: "clef-flash-9b")
            })
            let runner = RecipeRunner(service: service)
            return Store(initialState: state, run: runner.run)
        } catch {
            let message = "The recipe demo could not open its resources: " + error.localizedDescription
            var state = RecipeState(vocabulary: FoodVocabulary(foods: [], units: [:]))
            state.message = message
            let store = Store<RecipeFeature>(initialState: state) { effect, send in
                switch effect {
                case .waitForIdle(let session): send(.stopped(session))
                case .release(let session): send(.released(session))
                default: send(.failed(effect.session, message))
                }
            }
            return store
        }
    }

    static func initialState(managed: Bool = false) throws -> RecipeState {
        let path =
            managed ? "" : ProcessInfo.processInfo.environment["COREAI_DECISION_BUNDLE"] ?? ""
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
