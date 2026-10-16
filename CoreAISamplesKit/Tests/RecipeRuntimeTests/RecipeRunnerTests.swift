import FeatureStore
import Foundation
import RecipeCore
import RecipeFeature
import RecipeRuntime
import Testing

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct RecipeRunnerTests {
    @Test("Every service effect returns the matching success action and only successful loads save a path")
    func successes() async {
        let service = FakeRecipeService()
        var saved: [String] = []
        let runner = RecipeRunner(service: service) { saved.append($0) }
        let pairs: [(RecipeEffect, RecipeAction)] = [
            (.inspect(1, URL(filePath: "/photo.png")), .photoLoaded(1, service.photo)),
            (.fetch(2, "https://example.com"), .recipeLoaded(2, service.recipe)),
            (.loadModel(3, "/model"), .modelLoaded(3, "test-model")),
            (.check(4, 5, service.input), .checked(4, 5, service.report)),
            (.waitForIdle(6), .stopped(6)),
        ]
        for (effect, expected) in pairs {
            var actions: [RecipeAction] = []
            await runner.run(effect) { actions.append($0) }
            #expect(actions == [expected])
        }
        #expect(saved == ["/model"])
    }

    @Test("Service failures return errors while cancellation never publishes a failure or saves a model")
    func failures() async {
        let service = FakeRecipeService()
        var saved: [String] = []
        let runner = RecipeRunner(service: service) { saved.append($0) }
        let effects: [RecipeEffect] = [
            .inspect(1, URL(filePath: "/photo.png")), .fetch(1, "bad"), .loadModel(1, "/model"),
            .check(1, 2, service.input),
        ]
        for effect in effects {
            service.failure = RecipeError.invalid("Expected failure")
            var actions: [RecipeAction] = []
            await runner.run(effect) { actions.append($0) }
            #expect(actions == [.failed(1, "Expected failure")])
            service.failure = CancellationError()
            actions = []
            await runner.run(effect) { actions.append($0) }
            #expect(actions.isEmpty)
        }
        #expect(saved.isEmpty)
    }

    @Test(
        "The actual Store and Runner wait for cancelled work, including after closing and reopening",
        arguments: [false, true])
    func stopBarrier(reopen: Bool) async {
        let service = FakeRecipeService()
        service.suspend = true
        let runner = RecipeRunner(service: service)
        var initial = RecipeState(vocabulary: FoodVocabulary(foods: [], units: [:]))
        initial.modelName = "test-model"
        initial.recipe = service.recipe
        initial.inventoryComplete = true
        let completions = AsyncStream<Void>.makeStream()
        var finished = completions.stream.makeAsyncIterator()
        var events = service.events.stream.makeAsyncIterator()
        let store = Store<RecipeFeature>(initialState: initial) { effect, send in
            await runner.run(effect, send: send)
            completions.continuation.yield(())
        }
        store.send(.check)
        #expect(await events.next() == "check")
        store.send(.stop)
        #expect(await events.next() == "idle")
        if reopen {
            store.send(.ended)
            store.send(.appeared)
            #expect(await events.next() == "idle")
        }
        #expect(store.state.phase == .stopping && !store.state.canCheck)
        service.finish()
        while store.state.busy { _ = await finished.next() }
        #expect(service.sawCancellation)
        #expect(store.state.report == nil && store.state.canCheck)
        service.suspend = false
        store.send(.check)
        while store.state.busy { _ = await finished.next() }
        #expect(store.state.report == service.report)
        store.send(.ended)
    }
}

@MainActor
private final class FakeRecipeService: RecipeService {
    var failure: (any Error)?
    var suspend = false
    var sawCancellation = false
    let events = AsyncStream<String>.makeStream()
    private var pending: CheckedContinuation<Void, Never>?
    private var idleWaiters: [CheckedContinuation<Void, Never>] = []
    let photo = FridgePhoto(image: Data(), filename: "photo.png", ingredients: [], checks: [], seconds: 1)
    let recipe = Recipe(
        title: "Test", source: "https://example.com", servings: "2", duration: "", ingredients: [], steps: ["Cook."])
    let report = RecipeReport(
        rows: [], equipment: .available, equipmentProbability: 1, seconds: 1, reusedTokens: 0, modelVersion: "test")
    var input: RecipeCheckInput { RecipeCheckInput(recipe: recipe, pantry: [], inventoryComplete: true, kitchen: []) }

    func inspect(_ url: URL) async throws -> FridgePhoto {
        if let failure { throw failure }
        return photo
    }

    func fetch(_ url: String) async throws -> Recipe {
        if let failure { throw failure }
        return recipe
    }

    func loadModel(_ path: String) async throws -> String {
        if let failure { throw failure }
        return "test-model"
    }

    func check(_ input: RecipeCheckInput) async throws -> RecipeReport {
        if let failure { throw failure }
        events.continuation.yield("check")
        if suspend { await withCheckedContinuation { pending = $0 } }
        sawCancellation = Task.isCancelled
        for waiter in idleWaiters { waiter.resume() }
        idleWaiters = []
        return report
    }

    func waitForIdle() async {
        events.continuation.yield("idle")
        if pending != nil { await withCheckedContinuation { idleWaiters.append($0) } }
    }

    func finish() {
        pending?.resume()
        pending = nil
    }
}
