import FeatureStore
import Foundation
import ImageSearchFeature
import ImageSearchRuntime
import Testing

@Suite("Image search interaction contracts")
struct ImageSearchTests {
    let folder = URL(filePath: "/models/embedding")
    let photo = SearchPhoto(url: URL(filePath: "/images/photo.png"), thumbnail: Data())

    func ready() -> ImageSearchState {
        var state = ImageSearchState(modelFolder: folder)
        state.ready = true
        state.photos = [photo]
        state.query = "red box"
        return state
    }

    @Test("Reject missing models, empty queries, empty galleries, and duplicate starts")
    func guards() {
        var state = ImageSearchState()
        #expect(ImageSearchFeature.reduce(&state, .search).isEmpty)
        #expect(ImageSearchFeature.reduce(&state, .index([photo.url])).isEmpty)
        state = ready()
        state.query = " \n "
        #expect(ImageSearchFeature.reduce(&state, .search).isEmpty)
        state.query = "red box"
        #expect(!ImageSearchFeature.reduce(&state, .search).isEmpty)
        #expect(ImageSearchFeature.reduce(&state, .search).isEmpty)
        #expect(ImageSearchFeature.reduce(&state, .load(folder)).isEmpty)
        #expect(ImageSearchFeature.reduce(&state, .index([photo.url])).isEmpty)
    }

    @Test("Use a submitted query snapshot while preserving later draft edits")
    func snapshot() {
        var state = ready()
        #expect(ImageSearchFeature.reduce(&state, .search) == [.run(.search(1, "red box"), id: .work)])
        _ = ImageSearchFeature.reduce(&state, .setQuery("cheese"))
        let match = ImageSearchMatch(photo: photo, similarity: 0.3)
        _ = ImageSearchFeature.reduce(&state, .searched(1, [match], 0.12))
        #expect(state.submittedQuery == "red box")
        #expect(state.query == "cheese")
        #expect(state.matches == [match])
        #expect(state.canSearch)
    }

    @Test("A completed search with weak candidates is distinct from the initial gallery and can reveal all candidates")
    func weakMatches() {
        var state = ready()
        _ = ImageSearchFeature.reduce(&state, .search)
        let weak = ImageSearchMatch(photo: photo, similarity: 0.5)
        _ = ImageSearchFeature.reduce(&state, .searched(1, [weak], 0.1))
        #expect(state.hasSearched)
        #expect(state.displayedMatches.isEmpty)
        _ = ImageSearchFeature.reduce(&state, .showAllCandidates(true))
        #expect(state.displayedMatches == [weak])
        _ = ImageSearchFeature.reduce(&state, .index([photo.url]))
        #expect(!state.hasSearched && !state.showsAllCandidates)
        #expect(state.matches.isEmpty)
    }

    @Test("The fixed relevance cutoff retains its boundary and cancelled searches do not show an empty success")
    func relevanceBoundaryAndCancellation() {
        var state = ready()
        _ = ImageSearchFeature.reduce(&state, .search)
        let boundary = ImageSearchMatch(photo: photo, similarity: ImageSearchState.minimumSimilarity)
        _ = ImageSearchFeature.reduce(&state, .searched(1, [boundary], 0.1))
        #expect(state.displayedMatches == [boundary])
        _ = ImageSearchFeature.reduce(&state, .search)
        _ = ImageSearchFeature.reduce(&state, .stop)
        _ = ImageSearchFeature.reduce(&state, .searched(2, [], 0.1))
        _ = ImageSearchFeature.reduce(&state, .stopped(3))
        #expect(!state.hasSearched && state.matches.isEmpty)
    }

    @Test("Stop rejects late success and failure until the runtime becomes idle")
    func stop() {
        var state = ready()
        _ = ImageSearchFeature.reduce(&state, .search)
        #expect(ImageSearchFeature.reduce(&state, .stop) == [.cancel(.work), .run(.waitForIdle(2), id: .control)])
        _ = ImageSearchFeature.reduce(&state, .searched(1, [ImageSearchMatch(photo: photo, similarity: 1)], 1))
        _ = ImageSearchFeature.reduce(&state, .failed(1, ImageSearchFailure(message: "late")))
        #expect(state.matches.isEmpty && state.failure == nil)
        #expect(!state.canSearch)
        _ = ImageSearchFeature.reduce(&state, .stopped(1))
        #expect(state.phase == .stopping)
        _ = ImageSearchFeature.reduce(&state, .stopped(2))
        #expect(state.canSearch)
    }

    @Test("Closing during stop drains the matching generation and clears old model results")
    func closeAndReload() {
        var state = ready()
        _ = ImageSearchFeature.reduce(&state, .search)
        _ = ImageSearchFeature.reduce(&state, .stop)
        #expect(ImageSearchFeature.reduce(&state, .disappeared) == [.cancelAll, .run(.close(3), id: .control)])
        _ = ImageSearchFeature.reduce(&state, .stopped(2))
        #expect(state.isBusy)
        _ = ImageSearchFeature.reduce(&state, .stopped(3))
        #expect(state.photos.isEmpty && !state.ready && !state.isBusy)
        _ = ImageSearchFeature.reduce(&state, .load(folder))
        _ = ImageSearchFeature.reduce(&state, .loaded(1))
        #expect(!state.ready)
        _ = ImageSearchFeature.reduce(&state, .loaded(4))
        #expect(state.ready)
    }

    @Test("Indexing deduplicates files, validates limits, and permits retry after failure")
    func indexAndRetry() {
        var state = ready()
        #expect(
            ImageSearchFeature.reduce(&state, .index([photo.url, photo.url])) == [
                .run(.index(1, [photo.url]), id: .work)
            ])
        _ = ImageSearchFeature.reduce(&state, .failed(1, ImageSearchFailure(message: "invalid image")))
        #expect(!state.isBusy && state.ready && state.photos.isEmpty)
        #expect(!ImageSearchFeature.reduce(&state, .index([photo.url])).isEmpty)
        _ = ImageSearchFeature.reduce(&state, .indexed(2, [photo], 0.5))
        #expect(state.photos == [photo] && state.failure == nil)
        #expect(ImageSearchFeature.reduce(&state, .index(Array(repeating: photo.url, count: 51))).isEmpty)
        #expect(state.failure != nil && state.photos == [photo])
    }

    @MainActor
    @Test("The real Store and runner keep stop active until an in-flight service returns", .timeLimit(.minutes(1)))
    func storeCancellation() async {
        let service = SuspendedSearchService(photo: photo)
        let runner = ImageSearchRunner(service: service)
        let stopped = AsyncStream<Void>.makeStream()
        let store = Store<ImageSearchFeature>(initialState: ready()) { effect, send in
            await runner.run(effect) { action in
                send(action)
                if case .stopped = action { stopped.continuation.yield(()) }
            }
        }
        store.send(.search)
        var entered = service.entered.stream.makeAsyncIterator()
        _ = await entered.next()
        store.send(.stop)
        var waiting = service.waiting.stream.makeAsyncIterator()
        _ = await waiting.next()
        #expect(store.state.phase == .stopping)
        store.send(.search)
        #expect(service.calls == 1)
        service.release()
        var completed = stopped.stream.makeAsyncIterator()
        _ = await completed.next()
        #expect(store.state.phase == .idle)
        #expect(store.state.matches.isEmpty)
        #expect(store.state.canSearch)
    }
}

@MainActor
private final class SuspendedSearchService: ImageSearchService {
    let entered = AsyncStream<Void>.makeStream()
    let waiting = AsyncStream<Void>.makeStream()
    let photo: SearchPhoto
    var calls = 0
    var continuation: CheckedContinuation<Void, Never>?
    var idle: [CheckedContinuation<Void, Never>] = []
    init(photo: SearchPhoto) { self.photo = photo }
    func load(_ folder: URL) async throws {}
    func index(_ urls: [URL], progress: (Int) -> Void) async throws -> [SearchPhoto] { [photo] }
    func search(_ query: String) async throws -> [ImageSearchMatch] {
        calls += 1
        await withCheckedContinuation {
            continuation = $0
            entered.continuation.yield(())
        }
        let waiters = idle
        idle = []
        for waiter in waiters { waiter.resume() }
        return [ImageSearchMatch(photo: photo, similarity: 1)]
    }
    func waitForIdle() async {
        await withCheckedContinuation {
            idle.append($0)
            waiting.continuation.yield(())
        }
    }
    func close() async throws {}
    func release() {
        continuation?.resume()
        continuation = nil
    }
}
