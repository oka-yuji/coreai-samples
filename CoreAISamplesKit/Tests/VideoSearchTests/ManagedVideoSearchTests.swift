import FeatureStore
import Foundation
import Observation
import Testing
import VideoSearchFeature
import VideoSearchRuntime

@MainActor
@Suite("Managed video search contracts", .timeLimit(.minutes(1)))
struct ManagedVideoSearchTests {
    let fp16 = URL(filePath: "/models/fp16")
    let int4 = URL(filePath: "/models/int4")
    let video = URL(filePath: "/videos/clip.mov")
    let unavailable = "The selected model is not available right now. Try again."

    @Test("A new owner's wrapper keeps the video and search results without retaining the previous runtime")
    func keepsDataAcrossWrapperReplacement() async throws {
        let services = ServicePool(count: 2)
        let first = ManagedVideoSearch { services.next() }
        let original = first.store(folder: fp16, identity: "fp16", isPermitted: { true })
        original.send(.index(video))
        await settle(original)
        original.send(.setQuery("a dog"))
        original.send(.search)
        await settle(original)
        let results = original.state.matches
        try await first.release()
        await settle(original)
        let next = ManagedVideoSearch(retaining: first.state) { services.next() }
        let replacement = next.store(folder: fp16, identity: "fp16", isPermitted: { true })
        #expect(replacement !== original && !replacement.state.ready && replacement.state.needsIndex)
        #expect(replacement.state.video == video && replacement.state.matches == results && !results.isEmpty)
        #expect(replacement.state.query == "a dog" && services.all[1].calls.isEmpty)
        replacement.send(.setQuery("new query"))
        #expect(original.state.query == "a dog" && replacement.state.query == "new query")
    }

    @Test("VID-04 A closed gate or a replaced artifact binding rejects loading without touching the service")
    func gateRejection() async {
        let gate = TestGate()
        let services = ServicePool(count: 2)
        let search = ManagedVideoSearch { services.next() }
        let fp16Store = search.store(folder: fp16, identity: "fp16", isPermitted: { gate.isOpen })
        gate.isOpen = false
        fp16Store.send(.index(video))
        await settle(fp16Store)
        #expect(fp16Store.state.failure?.message == unavailable && services.all[0].calls.isEmpty)
        gate.isOpen = true
        let int4Store = search.store(folder: int4, identity: "int4", isPermitted: { gate.isOpen })
        fp16Store.send(.index(video))
        await settle(fp16Store)
        #expect(fp16Store.state.failure?.message == unavailable && services.all[0].calls.isEmpty)
        int4Store.send(.index(video))
        await settle(int4Store)
        #expect(services.all[1].calls == ["load int4", "index clip.mov"] && int4Store.state.hasValidIndex)
        #expect(int4Store.state.completed == 2 && int4Store.state.total == 2 && int4Store.state.duration == 2)
    }

    @Test(
        "VID-04 After switching from A to B and back to A, the first A store cannot run and no binding keeps a runtime"
    )
    func returnToFirstArtifact() async throws {
        let services = ServicePool(count: 3)
        let search = ManagedVideoSearch { services.next() }
        let first = search.store(folder: fp16, identity: "fp16", isPermitted: { true })
        first.send(.index(video))
        await settle(first)
        try await search.release()
        await settle(first)
        let middle = search.store(folder: int4, identity: "int4", isPermitted: { true })
        try await search.release()
        await settle(middle)
        let last = search.store(folder: fp16, identity: "fp16", isPermitted: { true })
        first.send(.index(video))
        await settle(first)
        #expect(first !== last && first.state.failure?.message == unavailable)
        #expect(services.all[0].calls.filter { !$0.hasPrefix("close") } == ["load fp16", "index clip.mov"])
        last.send(.index(video))
        await settle(last)
        try await search.release()
        await settle(last)
        #expect(services.all[2].calls == ["load fp16", "index clip.mov", "close", "closed"])
        #expect(services.all.allSatisfy { $0.calls.last == "closed" })
    }

    @Test(
        "VID-04 A release waits for the close in progress, throws its failure to every caller and the store, and a retry succeeds"
    )
    func closeFailure() async throws {
        let services = ServicePool(count: 1)
        let service = services.all[0]
        let search = ManagedVideoSearch { services.next() }
        let store = search.store(folder: fp16, identity: "fp16", isPermitted: { true })
        store.send(.index(video))
        await settle(store)
        store.send(.setQuery("a dog"))
        store.send(.search)
        await settle(store)
        let results = store.state.matches
        service.holdsClose = true
        service.failingCloses = [1]
        let stop = Task { try await search.release() }
        var closing = service.closing.stream.makeAsyncIterator()
        _ = await closing.next()
        let joined = Task { try await search.release() }
        service.finishClose()
        let error = await #expect(throws: VideoSearchReleaseError.self) { try await stop.value }
        await #expect(throws: VideoSearchReleaseError.self) { try await joined.value }
        await settle(store)
        #expect(error?.detail == "StubError()" && service.calls.filter { $0 == "close" }.count == 1)
        #expect(store.state.failure?.message == "The model could not be released." && !store.state.ready)
        #expect(store.state.video == video && store.state.matches == results && !results.isEmpty)
        service.holdsClose = false
        try await search.release()
        await settle(store)
        #expect(store.state.failure == nil && !store.state.isBusy && !store.state.ready && store.state.needsIndex)
        #expect(service.calls.filter { $0 == "close" }.count == 2)
    }

    @Test("VID-04 A video that cannot be read fails with its own message and keeps the demo usable")
    func invalidVideo() async throws {
        let services = ServicePool(count: 1)
        services.all[0].indexFailures = [VideoSearchError.invalidVideo]
        let search = ManagedVideoSearch { services.next() }
        let store = search.store(folder: fp16, identity: "fp16", isPermitted: { true })
        store.send(.index(video))
        await settle(store)
        #expect(store.state.failure?.message == "This video could not be read. Choose another video.")
        #expect(store.state.ready && store.state.canIndex && store.state.needsIndex && store.state.video == video)
        store.send(.index(video))
        await settle(store)
        #expect(store.state.failure == nil && store.state.hasValidIndex)
        try await search.release()
        await settle(store)
        #expect(services.all[0].calls.suffix(2) == ["close", "closed"])
    }
}

@MainActor
private func settle(_ store: Store<VideoSearchFeature>) async {
    for await state in Observations({ store.state }) where !state.isBusy { return }
}

private struct StubError: Error {}

@MainActor
private final class TestGate {
    var isOpen = true
}

@MainActor
private final class ServicePool {
    let all: [ScriptedVideoService]
    private(set) var made = 0

    init(count: Int) { all = (0..<count).map { _ in ScriptedVideoService() } }

    func next() -> any VideoSearchService {
        defer { made += 1 }
        return all[made]
    }
}

@MainActor
private final class ScriptedVideoService: VideoSearchService {
    var calls: [String] = []
    var indexFailures: [any Error] = []
    var failingCloses: Set<Int> = []
    var holdsClose = false
    let closing = AsyncStream<Void>.makeStream()
    private var heldCloses: [CheckedContinuation<Void, Never>] = []
    private var closeCount = 0
    private var indexed: [VideoMoment] = []

    func load(_ folder: URL) async throws { calls.append("load " + folder.lastPathComponent) }

    func index(_ video: URL, progress: (Int, Int) -> Void) async throws -> VideoIndex {
        calls.append("index " + video.lastPathComponent)
        if !indexFailures.isEmpty { throw indexFailures.removeFirst() }
        indexed = [0.5, 1.5].map { VideoMoment(time: $0, thumbnail: Data()) }
        progress(1, 2)
        progress(2, 2)
        return VideoIndex(duration: 2, moments: indexed)
    }

    func search(_ query: String) async throws -> [VideoSearchMatch] {
        calls.append("search " + query)
        return indexed.map { VideoSearchMatch(moment: $0, similarity: 0.8) }
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
