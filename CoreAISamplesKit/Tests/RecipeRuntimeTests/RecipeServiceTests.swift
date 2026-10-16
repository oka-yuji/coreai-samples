import Foundation
import RecipeCore
import Testing

@testable import RecipeRuntime

@Suite(.timeLimit(.minutes(1)))
@MainActor
struct RecipeServiceTests {
    @Test("Stopping waits for a cancelled photo operation to finish before the next operation starts")
    func stoppingBarrier() async throws {
        let photos = DeferredPhoto()
        let queued = AsyncStream<Void>.makeStream()
        let calls = RecipeCalls()
        let service = LiveRecipeService(
            vocabulary: try FoodVocabulary.bundled(),
            inspectPhoto: { _ in await photos.read() },
            getRecipe: { _ in await calls.fetch() },
            onQueued: { queued.continuation.yield(()) })
        let operation = Task { try await service.inspect(URL(filePath: "/unused.png")) }
        await photos.waitUntilStarted()
        var arrivals = queued.stream.makeAsyncIterator()
        let stopping = Task { await service.waitForIdle() }
        _ = await arrivals.next()
        let following = Task { try await service.fetch("unused") }
        _ = await arrivals.next()
        #expect(await calls.count == 0)
        operation.cancel()
        await photos.finish()
        await #expect(throws: CancellationError.self) { try await operation.value }
        await stopping.value
        #expect(try await following.value.title == "After stop")
        #expect(await calls.count == 1)
        queued.continuation.finish()
    }

    @Test("A failed operation releases the service for a later request")
    func failureRecovery() async throws {
        let calls = RecipeCalls()
        let service = LiveRecipeService(
            vocabulary: try FoodVocabulary.bundled(),
            inspectPhoto: { _ in throw RecipeError.invalid("Unreadable photo") },
            getRecipe: { _ in await calls.fetch() })
        await #expect(throws: RecipeError.self) { try await service.inspect(URL(filePath: "/unused.png")) }
        #expect(try await service.fetch("unused").title == "After stop")
    }
}

private actor DeferredPhoto {
    private var continuation: CheckedContinuation<FridgePhoto, Never>?
    private var arrival: CheckedContinuation<Void, Never>?

    func read() async -> FridgePhoto {
        await withCheckedContinuation {
            continuation = $0
            arrival?.resume()
            arrival = nil
        }
    }

    func waitUntilStarted() async {
        if continuation == nil { await withCheckedContinuation { arrival = $0 } }
    }

    func finish() {
        continuation?.resume(
            returning: FridgePhoto(image: Data(), filename: "", ingredients: [], checks: [], seconds: 0))
        continuation = nil
    }
}

private actor RecipeCalls {
    private(set) var count = 0

    func fetch() -> Recipe {
        count += 1
        return Recipe(title: "After stop", source: "", servings: "", duration: "", ingredients: [], steps: [])
    }
}
