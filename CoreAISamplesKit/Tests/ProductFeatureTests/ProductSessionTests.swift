import FeatureStore
import Foundation
import ProductCore
import ProductFeature
import Testing

@testable import ProductRuntime

struct ProductSessionTests {
    private let stamp = Date(timeIntervalSince1970: 0)
    private let selected = URL(filePath: "/models/selected")
    private let identity = "gliner2.5-multi/gliner2.5-multi-fp32@0123456789abcdef0123456789abcdef01234567"

    private func managed() -> ProductState {
        var state = ProductState(folder: selected, resourceIdentity: identity)
        state.draft.mode = .text
        state.draft.text = "Memory: 16 GB"
        return state
    }

    private func source() throws -> ProductSource {
        try ProductSource(title: "Example", text: "Memory: 16 GB", url: nil, capturedAt: Date(timeIntervalSince1970: 0))
    }

    @MainActor
    @Test(
        "Imports and extracts only with the selected context and never calls the service for a missing, other-revision or old-session request"
    )
    func selectedContext() async throws {
        var state = managed()
        let unchanged = state
        #expect(ProductFeature.reduce(&state, .modelChosen(URL(filePath: "/models/other"))).isEmpty)
        #expect(state == unchanged)
        let gate = ProductGate()
        let service = ProductServiceFake()
        let runner = ProductRunner(service: service) { gate.open }
        let actions = ProductActions()
        await runner.run(.load(1, state.draft, stamp), send: actions.append)
        await runner.run(.extract(1, try source(), selected), send: actions.append)
        #expect(service.log.isEmpty)
        #expect(actions.failures == [1, 1])
        gate.open = true
        #expect(ProductFeature.reduce(&state, .submit(stamp)) == [.run(.load(1, state.draft, stamp), id: .operation)])
        await runner.run(.load(1, state.draft, stamp), send: actions.append)
        let loaded = try #require(actions.sources.first)
        #expect(
            ProductFeature.reduce(&state, .sourceLoaded(1, loaded)) == [
                .run(.extract(1, loaded, selected), id: .operation)
            ])
        await runner.run(.extract(1, loaded, selected), send: actions.append)
        _ = ProductFeature.reduce(&state, .extracted(1, try #require(actions.memos.first)))
        #expect(service.log == ["source", "source finished", "extract", "extract finished"])
        #expect(service.folders == [selected])
        #expect(state.cards.map(\.resourceIdentity) == [identity])
    }

    @MainActor
    @Test(
        "Going back, stopping or re-entering during a fetch or extraction waits for the work to end and the model to be released, and a late result never reaches the next screen",
        .timeLimit(.minutes(1)), arguments: ProductStage.allCases)
    func stopDuringWork(stage: ProductStage) async throws {
        let service = ProductServiceFake(holding: true)
        let runner = ProductRunner(service: service)
        var state = managed()
        state.cards = [ProductCard(id: 0, memo: try ProductMemo(source: source(), candidates: []))]
        let store = Store<ProductFeature>(initialState: state, run: runner.run)
        var events = service.events.makeAsyncIterator()
        store.send(.submit(stamp))
        try await receive(&events, "source")
        var work = ["source"]
        if stage == .extraction {
            service.resume()
            try await receive(&events, "source finished", "extract")
            #expect(store.state.phase == .extracting)
            work += ["source finished", "extract"]
        }
        store.send(.closed)
        let stopping = Task { try await runner.release() }
        try await receive(&events, "waiting", "waiting")
        store.send(.appeared)
        #expect(!store.state.canAdd)
        service.resume()
        try await stopping.value
        try await receive(&events, "release", "release")
        let finished = stage == .fetch ? "source finished" : "extract finished"
        #expect(service.log == work + ["waiting", "waiting", finished, "release", "release"])
        #expect(store.state.cards.map(\.id) == [0])
        #expect(store.state.importedSource == nil)
        #expect(store.state.phase == .idle && store.state.canAdd)
    }

    @MainActor
    @Test(
        "A release failure reaches the stop request and the feature, while a cancelled stop still completes and a cancelled drain reports nothing",
        .timeLimit(.minutes(1)))
    func releaseFailure() async throws {
        let service = ProductServiceFake()
        service.releaseFailure = .busy
        let runner = ProductRunner(service: service)
        await #expect(throws: ProductError.busy) { try await runner.release() }
        let actions = ProductActions()
        await runner.run(.release(3), send: actions.append)
        #expect(actions.releaseFailures == [3])
        let failing = Task { await runner.run(.release(4), send: actions.append) }
        failing.cancel()
        await failing.value
        service.releaseFailure = nil
        let releasing = Task { await runner.run(.release(5), send: actions.append) }
        releasing.cancel()
        await releasing.value
        let stopping = Task { try await runner.release() }
        stopping.cancel()
        try await stopping.value
        #expect(actions.all.count == 1)
        #expect(service.log == ["release", "release", "release", "release", "release"])
    }

    @MainActor
    @Test(
        "The live service refuses to release during a page read and releases after the read ends",
        .timeLimit(.minutes(1)))
    func liveRelease() async throws {
        let (started, starting) = AsyncStream.makeStream(of: Void.self)
        let (proceed, proceeding) = AsyncStream.makeStream(of: Void.self)
        let page = ProductPageClient { _ in
            starting.yield()
            for await _ in proceed { break }
            return ProductHTTPPage(status: 200, location: nil, mime: "text/plain", data: Data("Memory 16 GB".utf8))
        }
        let service = LiveProductService(page: page)
        var draft = ProductDraft()
        draft.address = "https://www.apple.com/macbook-air/specs/"
        let request = draft
        let capturedAt = stamp
        let reading = Task { try await service.source(draft: request, capturedAt: capturedAt) }
        var starts = started.makeAsyncIterator()
        _ = await starts.next()
        await #expect(throws: ProductError.busy) { try await service.release() }
        let releasing = Task { try await ProductRunner(service: service).release() }
        proceeding.yield()
        #expect(try await reading.value.text == "Memory 16 GB")
        try await releasing.value
        try await service.release()
    }
}

enum ProductStage: CaseIterable, Sendable {
    case fetch, extraction
}

private struct MissingEvents: Error {
    let events: [String]
}

@MainActor
private func receive(_ events: inout AsyncStream<String>.Iterator, _ wanted: String...) async throws {
    var remaining = wanted
    while !remaining.isEmpty, let event = await events.next(isolation: #isolation) {
        if let index = remaining.firstIndex(of: event) { remaining.remove(at: index) }
    }
    guard remaining.isEmpty else { throw MissingEvents(events: remaining) }
}

@MainActor
private final class ProductGate {
    var open = false
}

@MainActor
private final class ProductActions {
    private(set) var all: [ProductAction] = []

    func append(_ action: ProductAction) { all.append(action) }

    var failures: [Int] {
        all.compactMap { action -> Int? in if case .failed(let id, _) = action { id } else { nil } }
    }

    var releaseFailures: [Int] {
        all.compactMap { action -> Int? in if case .releaseFailed(let id, _) = action { id } else { nil } }
    }

    var sources: [ProductSource] {
        all.compactMap { action -> ProductSource? in
            if case .sourceLoaded(_, let source) = action { source } else { nil }
        }
    }

    var memos: [ProductMemo] {
        all.compactMap { action -> ProductMemo? in if case .extracted(_, let memo) = action { memo } else { nil } }
    }
}

@MainActor
private final class ProductServiceFake: ProductService {
    let events: AsyncStream<String>
    private let sink: AsyncStream<String>.Continuation
    private let holding: Bool
    private(set) var log: [String] = []
    private(set) var folders: [URL] = []
    var releaseFailure: ProductError?
    private var held: [CheckedContinuation<Void, Never>] = []
    private var running = 0
    private var idle: [CheckedContinuation<Void, Never>] = []

    init(holding: Bool = false) {
        self.holding = holding
        (events, sink) = AsyncStream.makeStream(of: String.self)
    }

    func source(draft: ProductDraft, capturedAt: Date) async throws -> ProductSource {
        await perform("source")
        return try ProductSource(title: draft.title, text: draft.text, url: nil, capturedAt: capturedAt)
    }

    func extract(source: ProductSource, folder: URL) async throws -> ProductMemo {
        folders.append(folder)
        await perform("extract")
        return try ProductMemo(source: source, candidates: [])
    }

    func waitForIdle() async {
        guard running > 0 else { return }
        record("waiting")
        await withCheckedContinuation { idle.append($0) }
    }

    func release() async throws {
        record("release")
        if let releaseFailure { throw releaseFailure }
    }

    func resume() {
        guard !held.isEmpty else { return }
        held.removeFirst().resume()
    }

    private func perform(_ name: String) async {
        running += 1
        record(name)
        if holding { await withCheckedContinuation { held.append($0) } }
        record(name + " finished")
        running -= 1
        guard running == 0 else { return }
        let waiting = idle
        idle.removeAll()
        for continuation in waiting { continuation.resume() }
    }

    private func record(_ event: String) {
        log.append(event)
        sink.yield(event)
    }
}
