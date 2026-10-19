import FeatureStore
import Foundation
import ImageSearchFeature
import ImageSearchRuntime
import Observation
import Testing

@MainActor
@Suite("Managed image search contracts", .timeLimit(.minutes(1)))
struct ManagedImageSearchTests {
    let fp16 = URL(filePath: "/models/fp16")
    let int4 = URL(filePath: "/models/int4")
    let images = [URL(filePath: "/images/a.png"), URL(filePath: "/images/b.png")]
    let unavailable = "The selected model is not available right now. Try again."

    @Test("IMAGE-01 A managed index loads only the selected artifact first and rejects another model folder")
    func selectedArtifactOnly() {
        var state = ImageSearchState(managed: fp16, identity: "fp16")
        #expect(ImageSearchFeature.reduce(&state, .load(int4)).isEmpty)
        #expect(state.modelFolder == fp16 && !state.isBusy)
        #expect(ImageSearchFeature.reduce(&state, .index(images + images)) == [.run(.load(1, fp16), id: .work)])
        #expect(ImageSearchFeature.reduce(&state, .loaded(1)) == [.run(.index(1, images), id: .work)])
        #expect(state.ready && state.phase == .indexing && state.total == 2)
    }

    @Test("IMAGE-01 An image selection that arrives after its session was stopped is rejected")
    func lateImageSelection() {
        var state = ImageSearchState(managed: fp16, identity: "fp16")
        let request = state.session
        _ = ImageSearchFeature.reduce(&state, .closed)
        let current = state.session
        _ = ImageSearchFeature.reduce(&state, .stopped(current))
        #expect(ImageSearchFeature.reduce(&state, .imagesChosen(request, images)).isEmpty)
        #expect(state.photos.isEmpty && !state.isBusy)
        #expect(ImageSearchFeature.reduce(&state, .imagesChosen(current, images)) == [.run(.load(2, fp16), id: .work)])
    }

    @Test("IMAGE-01 A closed gate or a replaced artifact binding rejects loading without touching the service")
    func gateRejection() async {
        let gate = TestGate()
        let services = ServicePool(count: 2)
        let search = ManagedImageSearch { services.next() }
        let fp16Store = search.store(folder: fp16, identity: "fp16", isPermitted: { gate.isOpen })
        gate.isOpen = false
        fp16Store.send(.index(images))
        await settle(fp16Store)
        #expect(fp16Store.state.failure?.message == unavailable && services.all[0].calls.isEmpty)
        gate.isOpen = true
        let int4Store = search.store(folder: int4, identity: "int4", isPermitted: { gate.isOpen })
        fp16Store.send(.index(images))
        await settle(fp16Store)
        #expect(fp16Store.state.failure?.message == unavailable && services.all[0].calls.isEmpty)
        int4Store.send(.index(images))
        await settle(int4Store)
        #expect(services.all[1].calls == ["load int4", "index 2"] && int4Store.state.hasValidIndex)
    }

    @Test(
        "IMAGE-01 After switching from A to B and back to A, the first A store cannot run and no binding keeps a runtime"
    )
    func returnToFirstArtifact() async throws {
        let services = ServicePool(count: 4)
        let search = ManagedImageSearch { services.next() }
        let first = search.store(folder: fp16, identity: "fp16", isPermitted: { true })
        first.send(.index(images))
        await settle(first)
        try await search.release()
        await settle(first)
        let middle = search.store(folder: int4, identity: "int4", isPermitted: { true })
        try await search.release()
        await settle(middle)
        let last = search.store(folder: fp16, identity: "fp16", isPermitted: { true })
        first.send(.index(images))
        await settle(first)
        #expect(first !== last && first.state.failure?.message == unavailable)
        #expect(services.all[0].calls.filter { !$0.hasPrefix("close") } == ["load fp16", "index 2"])
        last.send(.index(images))
        await settle(last)
        let replacement = search.store(folder: int4, identity: "int4", isPermitted: { true })
        try await search.release()
        await settle(replacement)
        #expect(services.all[2].calls == ["load fp16", "index 2", "close", "closed"])
        #expect(services.all.allSatisfy { $0.calls.last == "closed" })
    }

    @Test("IMAGE-01 A file selection failure that arrives after its session was stopped is ignored")
    func lateSelectionFailure() {
        var state = ImageSearchState(managed: fp16, identity: "fp16")
        let request = state.session
        _ = ImageSearchFeature.reduce(&state, .closed)
        let current = state.session
        _ = ImageSearchFeature.reduce(&state, .stopped(current))
        _ = ImageSearchFeature.reduce(&state, .selectionFailed(request, "old callback"))
        #expect(state.failure == nil)
        _ = ImageSearchFeature.reduce(&state, .selectionFailed(current, "current callback"))
        #expect(state.failure?.detail == "current callback")
    }

    @Test("IMAGE-02 Stopping releases the runtime and re-entering the artifact keeps the images, query, and results")
    func retainedAcrossStop() async throws {
        let services = ServicePool(count: 1)
        let search = ManagedImageSearch { services.next() }
        let store = search.store(folder: fp16, identity: "fp16", isPermitted: { true })
        store.send(.index(images))
        await settle(store)
        store.send(.setQuery("red box"))
        store.send(.search)
        await settle(store)
        store.send(.showAllCandidates(true))
        store.send(.select(store.state.photos.first))
        let results = store.state.matches
        try await search.release()
        await settle(store)
        let state = store.state
        #expect(services.all[0].calls.contains("close") && !state.ready && !state.hasValidIndex)
        #expect(state.photos.map(\.url) == images && state.query == "red box" && state.submittedQuery == "red box")
        #expect(state.matches == results && !results.isEmpty && state.showsAllCandidates && state.selection == nil)
        #expect(!state.canSearch && state.needsIndex && state.canIndex)
        let again = search.store(folder: fp16, identity: "fp16", isPermitted: { true })
        #expect(again === store && search.state?.matches == results)
        again.send(.index(state.photos.map(\.url)))
        await settle(again)
        #expect(again.state.hasValidIndex && services.all[0].calls.suffix(2) == ["load fp16", "index 2"])
    }

    @Test("IMAGE-02 A standalone screen shown again re-indexes its kept images by loading the same model folder first")
    func standaloneReentry() {
        let photos = images.map { SearchPhoto(url: $0, thumbnail: Data()) }
        let results = [ImageSearchMatch(photo: photos[0], similarity: 0.8)]
        var state = ImageSearchState(modelFolder: fp16)
        state.ready = true
        state.photos = photos
        state.indexIdentity = state.currentIndexIdentity
        state.query = "red box"
        state.matches = results
        state.hasSearched = true
        _ = ImageSearchFeature.reduce(&state, .disappeared)
        let closed = state.session
        _ = ImageSearchFeature.reduce(&state, .stopped(closed))
        #expect(state.photos == photos && state.matches == results && state.query == "red box")
        #expect(state.canIndex && state.needsIndex && !state.canSearch)
        var reloaded = state
        _ = ImageSearchFeature.reduce(&reloaded, .load(fp16))
        #expect(reloaded.photos == photos && reloaded.matches == results && !reloaded.hasValidIndex)
        let reload = reloaded.session
        _ = ImageSearchFeature.reduce(&reloaded, .loaded(reload))
        #expect(reloaded.ready && !reloaded.hasValidIndex && reloaded.needsIndex)
        _ = ImageSearchFeature.reduce(&reloaded, .load(int4))
        #expect(reloaded.photos.isEmpty && reloaded.matches.isEmpty)
        #expect(ImageSearchFeature.reduce(&state, .index(images)) == [.run(.load(closed + 1, fp16), id: .work)])
        #expect(ImageSearchFeature.reduce(&state, .loaded(closed + 1)) == [.run(.index(closed + 1, images), id: .work)])
        #expect(state.photos == photos && state.phase == .indexing)
        _ = ImageSearchFeature.reduce(&state, .indexed(closed + 1, photos, 0.1))
        #expect(state.hasValidIndex && state.canSearch)
    }

    @Test("IMAGE-03 Another artifact re-indexes the retained images itself and labels each result with its artifact")
    func variantSwitch() async throws {
        let services = ServicePool(count: 2)
        let search = ManagedImageSearch { services.next() }
        let fp16Store = search.store(folder: fp16, identity: "fp16", isPermitted: { true })
        fp16Store.send(.index(images))
        await settle(fp16Store)
        fp16Store.send(.setQuery("red box"))
        fp16Store.send(.search)
        await settle(fp16Store)
        #expect(fp16Store.state.resultIdentity == "fp16")
        try await search.release()
        await settle(fp16Store)
        let int4Store = search.store(folder: int4, identity: "int4", isPermitted: { true })
        let carried = int4Store.state
        #expect(int4Store !== fp16Store && carried.resourceIdentity == "int4" && carried.modelFolder == int4)
        #expect(carried.photos == fp16Store.state.photos && carried.matches == fp16Store.state.matches)
        #expect(carried.resultIdentity == "fp16" && !carried.hasValidIndex && !carried.canSearch)
        int4Store.send(.search)
        #expect(!int4Store.state.isBusy && services.all[1].calls.isEmpty)
        int4Store.send(.index(carried.photos.map(\.url)))
        await settle(int4Store)
        int4Store.send(.search)
        await settle(int4Store)
        #expect(services.all[1].calls == ["load int4", "index 2", "search red box"])
        #expect(int4Store.state.resultIdentity == "int4" && int4Store.state.submittedQuery == "red box")
        #expect(services.all[0].calls.filter { $0.hasPrefix("load") } == ["load fp16"])
    }

    @Test("IMAGE-04 A failed release reaches the store and every stop caller as a close failure until a retry succeeds")
    func closeFailure() async throws {
        let services = ServicePool(count: 1)
        let service = services.all[0]
        let search = ManagedImageSearch { services.next() }
        let store = search.store(folder: fp16, identity: "fp16", isPermitted: { true })
        store.send(.index(images))
        await settle(store)
        service.holdsClose = true
        service.failingCloses = [1]
        let stop = Task { try await search.release() }
        var closing = service.closing.stream.makeAsyncIterator()
        _ = await closing.next()
        service.finishClose()
        let error = await #expect(throws: ImageSearchReleaseError.self) { try await stop.value }
        await settle(store)
        #expect(error?.detail == "StubError()" && service.calls.filter { $0 == "close" }.count == 1)
        #expect(store.state.failure?.message == "The model could not be released." && !store.state.ready)
        #expect(store.state.photos.count == 2)
        service.holdsClose = false
        try await search.release()
        await settle(store)
        #expect(store.state.failure == nil && !store.state.isBusy && !store.state.ready)
    }

    @Test(
        "IMAGE-04 A release that completes at once gives the stop caller and the store the same result from one close",
        arguments: [Set([1]), Set([2])])
    func immediateRelease(failing: Set<Int>) async {
        let services = ServicePool(count: 1)
        let service = services.all[0]
        service.failingCloses = failing
        let search = ManagedImageSearch { services.next() }
        let store = search.store(folder: fp16, identity: "fp16", isPermitted: { true })
        let stopFailed = (try? await search.release()) == nil
        await settle(store)
        #expect(service.calls.filter { $0 == "close" }.count == 1)
        #expect(stopFailed == failing.contains(1) && (store.state.failure != nil) == stopFailed)
    }

    @Test("IMAGE-04 A release from a cancelled task or before any content still completes and waits for the close")
    func cancelledRelease() async throws {
        let services = ServicePool(count: 1)
        let service = services.all[0]
        let search = ManagedImageSearch { services.next() }
        try await search.release()
        #expect(services.made == 0)
        let store = search.store(folder: fp16, identity: "fp16", isPermitted: { true })
        store.send(.index(images))
        await settle(store)
        service.holdsClose = true
        let stop = Task {
            try await search.release()
            service.calls.append("stopped")
        }
        stop.cancel()
        var closing = service.closing.stream.makeAsyncIterator()
        _ = await closing.next()
        service.finishClose()
        try await stop.value
        await settle(store)
        #expect(service.calls.suffix(3) == ["close", "closed", "stopped"])
        #expect(!store.state.ready && store.state.failure == nil && store.state.photos.count == 2)
    }

    @Test("IMAGE-04 Stopping during indexing rejects late progress and keeps the previous images for another index")
    func cancelDuringIndexing() {
        var state = ImageSearchState(managed: fp16, identity: "fp16")
        state.ready = true
        state.photos = [SearchPhoto(url: images[0], thumbnail: Data())]
        state.indexIdentity = state.currentIndexIdentity
        _ = ImageSearchFeature.reduce(&state, .index(images))
        _ = ImageSearchFeature.reduce(&state, .progress(1, 1))
        #expect(ImageSearchFeature.reduce(&state, .closed) == [.cancelAll, .run(.close(2), id: .control)])
        _ = ImageSearchFeature.reduce(&state, .progress(1, 2))
        _ = ImageSearchFeature.reduce(&state, .indexed(1, images.map { SearchPhoto(url: $0, thumbnail: Data()) }, 1))
        _ = ImageSearchFeature.reduce(&state, .stopped(2))
        #expect(state.photos.map(\.url) == [images[0]] && state.completed == 1 && !state.hasValidIndex)
        #expect(state.canIndex && state.needsIndex && state.failure == nil)
        #expect(ImageSearchFeature.reduce(&state, .index([images[0]])) == [.run(.load(3, fp16), id: .work)])
    }

    @Test("IMAGE-04 Invalid images and expired image URLs fail with their own messages and keep the demo usable")
    func imageFailures() async throws {
        var state = ImageSearchState(managed: fp16, identity: "fp16")
        let invalid = ImageSearchFailure(message: "Use an opaque, still image with no more than 40 million pixels.")
        _ = ImageSearchFeature.reduce(&state, .index(images))
        _ = ImageSearchFeature.reduce(&state, .failed(1, invalid))
        #expect(!state.isBusy && state.canIndex && state.failure == invalid)
        let services = ServicePool(count: 1)
        services.all[0].indexFailures = [CocoaError(.fileReadNoSuchFile, userInfo: [NSURLErrorKey: images[1]])]
        let search = ManagedImageSearch { services.next() }
        let store = search.store(folder: fp16, identity: "fp16", isPermitted: { true })
        store.send(.index(images))
        await settle(store)
        #expect(store.state.failure?.message == "Some images can no longer be opened. Choose the images again.")
        #expect(store.state.ready && store.state.canIndex && store.state.photos.isEmpty)
        try await search.release()
        await settle(store)
        #expect(services.all[0].calls.contains("close") && store.state.failure == nil)
        let missing = FileManager.default.temporaryDirectory.appending(path: "missing-\(UUID().uuidString).png")
        let error = await #expect(throws: CocoaError.self) {
            try await LiveImageSearchService().index([missing]) { _ in }
        }
        #expect(error?.code == .fileReadNoSuchFile)
    }
}

@MainActor
private func settle(_ store: Store<ImageSearchFeature>) async {
    for await state in Observations({ store.state }) where !state.isBusy { return }
}

private struct StubError: Error {}

@MainActor
private final class TestGate {
    var isOpen = true
}

@MainActor
private final class ServicePool {
    let all: [ScriptedSearchService]
    private(set) var made = 0

    init(count: Int) { all = (0..<count).map { _ in ScriptedSearchService() } }

    func next() -> any ImageSearchService {
        defer { made += 1 }
        return all[made]
    }
}

@MainActor
private final class ScriptedSearchService: ImageSearchService {
    var calls: [String] = []
    var indexFailures: [any Error] = []
    var failingCloses: Set<Int> = []
    var holdsClose = false
    let closing = AsyncStream<Void>.makeStream()
    private var heldCloses: [CheckedContinuation<Void, Never>] = []
    private var closeCount = 0
    private var indexed: [SearchPhoto] = []

    func load(_ folder: URL) async throws { calls.append("load " + folder.lastPathComponent) }

    func index(_ urls: [URL], progress: (Int) -> Void) async throws -> [SearchPhoto] {
        calls.append("index \(urls.count)")
        if !indexFailures.isEmpty { throw indexFailures.removeFirst() }
        progress(urls.count)
        indexed = urls.map { SearchPhoto(url: $0, thumbnail: Data()) }
        return indexed
    }

    func search(_ query: String) async throws -> [ImageSearchMatch] {
        calls.append("search " + query)
        return indexed.map { ImageSearchMatch(photo: $0, similarity: 0.8) }
    }

    func waitForIdle() async {}

    func close() async throws {
        closeCount += 1
        let number = closeCount
        calls.append("close")
        closing.continuation.yield(())
        if holdsClose { await withCheckedContinuation { heldCloses.append($0) } }
        calls.append("closed")
        if failingCloses.contains(number) { throw StubError() }
    }

    func finishClose() {
        let held = heldCloses
        heldCloses = []
        for continuation in held { continuation.resume() }
    }
}
