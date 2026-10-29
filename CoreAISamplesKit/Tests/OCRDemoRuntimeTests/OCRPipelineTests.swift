import FeatureStore
import Foundation
import OCRCore
import OCRFeature
import OCRRuntime
import Testing

@testable import OCRDemoRuntime

@Suite(.timeLimit(.minutes(1)))
struct OCRPipelineTests {
    @MainActor
    @Test("The real Store and runner remain stopping until the pipeline joins a cancellation-ignoring page")
    func storeStopAndRetry() async throws {
        let fixture = try pageFixture()
        let model = ControlledPage(document: fixture.document)
        let releaseEvents = AsyncStream<Void>.makeStream()
        let pipeline = OCRPipeline(loader: { _ in model }, onReleaseWaiting: { releaseEvents.continuation.yield(()) })
        let runner = OCRRunner(service: LiveOCRService(pipeline: pipeline))
        let delivered = AsyncStream<OCRAction>.makeStream()
        let store = Store<OCRFeature>(
            initialState: OCRState(bundle: URL(filePath: "/fixture/model"), input: fixture.input)
        ) {
            effect, send in
            await runner.run(effect) {
                send($0)
                delivered.continuation.yield($0)
            }
        }
        store.send(.appeared)
        store.send(.read)
        var entered = model.entered.makeAsyncIterator()
        _ = await entered.next()
        store.send(.stop)
        var waiting = releaseEvents.stream.makeAsyncIterator()
        _ = await waiting.next()
        #expect(store.state.phase == .stopping && !store.state.canRead)
        await model.complete()
        var actions = delivered.stream.makeAsyncIterator()
        while let action = await actions.next() {
            if case .stopped(2) = action { break }
        }
        #expect(store.state.phase == .cancelled && store.state.document == nil)
        store.send(.read)
        while let action = await actions.next() {
            if case .completed(3, _) = action { break }
        }
        #expect(store.state.phase == .complete && store.state.document == fixture.document)
        store.send(.ended)
        while let action = await actions.next() {
            if case .stopped(4) = action { break }
        }
        #expect(!store.state.presented && store.state.phase == .idle)
        delivered.continuation.finish()
        releaseEvents.continuation.finish()
    }

    @Test("Image selection preserves decoded RGB and hashes while creating a decodable preview")
    func imageSelection() async throws {
        let data = try #require(
            Data(
                base64Encoded:
                    "iVBORw0KGgoAAAANSUhEUgAAAAUAAAADCAIAAADUVFKvAAAAEElEQVR4nGOQdjiJjBgI8AGQrREdJlaCyAAAAABJRU5ErkJggg=="
            ))
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appending(path: "page.png")
        try data.write(to: file)
        let pipeline = OCRPipeline()
        let selected = try await pipeline.image(at: file)
        #expect(selected.name == "page.png" && selected.image.width == 5 && selected.image.height == 3)
        #expect(selected.sha256 == "10894818d97628c23f9ab0d2fbf6fc549302c34f7509bd11a6fa5dd0827eecef")
        #expect(selected.image.rgb == Array(repeating: [UInt8(27), 64, 201], count: 15).flatMap { $0 })
        #expect(try OCRImage(data: selected.preview) == selected.image)
        await pipeline.release()
    }

    @Test("Release joins a cancelled read before unloading and permits a fresh read afterwards")
    func releaseDuringRead() async throws {
        let fixture = try pageFixture()
        let model = ControlledPage(document: fixture.document)
        let releaseEvents = AsyncStream<Void>.makeStream()
        let pipeline = OCRPipeline(loader: { _ in model }, onReleaseWaiting: { releaseEvents.continuation.yield(()) })
        let bundle = URL(filePath: "/fixture/model")
        try await pipeline.load(bundle: bundle)
        let reading = Task { try await pipeline.read(fixture.input) }
        var entered = model.entered.makeAsyncIterator()
        _ = await entered.next()
        let releasing = Task { await pipeline.release() }
        var waiting = releaseEvents.stream.makeAsyncIterator()
        _ = await waiting.next()
        await #expect(throws: OCRRuntimeError.busy) { try await pipeline.load(bundle: bundle) }
        reading.cancel()
        await model.complete()
        await #expect(throws: CancellationError.self) { try await reading.value }
        await releasing.value
        await #expect(throws: OCRRuntimeError.invalidRequest("Load the model before reading an image.")) {
            try await pipeline.read(fixture.input)
        }
        try await pipeline.load(bundle: bundle)
        #expect(try await pipeline.read(fixture.input) == fixture.document)
        await pipeline.release()
        releaseEvents.continuation.finish()
    }

    @Test("Two releases join an unfinished load and a cancelled release cannot clear a later model")
    func releaseDuringLoad() async throws {
        let fixture = try pageFixture()
        let model = ControlledPage(document: fixture.document)
        await model.complete()
        let loader = ControlledLoader(model: model)
        let releaseEvents = AsyncStream<Void>.makeStream()
        let pipeline = OCRPipeline(
            loader: { _ in await loader.load() }, onReleaseWaiting: { releaseEvents.continuation.yield(()) })
        let bundle = URL(filePath: "/fixture/model")
        let loading = Task { try await pipeline.load(bundle: bundle) }
        var entered = loader.entered.makeAsyncIterator()
        _ = await entered.next()
        let first = Task { await pipeline.release() }
        let second = Task { await pipeline.release() }
        var waiting = releaseEvents.stream.makeAsyncIterator()
        _ = await waiting.next()
        _ = await waiting.next()
        first.cancel()
        loading.cancel()
        await loader.complete()
        await #expect(throws: CancellationError.self) { try await loading.value }
        await second.value
        try await pipeline.load(bundle: bundle)
        await first.value
        #expect(try await pipeline.read(fixture.input) == fixture.document)
        await pipeline.release()
        releaseEvents.continuation.finish()
    }

    @Test("Switching a loaded model requires release and invalid images do not leave the pipeline busy")
    func boundaryFailures() async throws {
        let fixture = try pageFixture()
        let model = ControlledPage(document: fixture.document)
        await model.complete()
        let pipeline = OCRPipeline(loader: { _ in model })
        let bundle = URL(filePath: "/fixture/model")
        try await pipeline.load(bundle: bundle)
        await #expect(throws: OCRRuntimeError.invalidRequest("Stop before changing the model.")) {
            try await pipeline.load(bundle: bundle.appending(path: "other"))
        }
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appending(path: "invalid.png")
        try Data("not an image".utf8).write(to: file)
        await #expect(throws: OCRError.invalidImage) { try await pipeline.image(at: file) }
        #expect(try await pipeline.read(fixture.input) == fixture.document)
        await pipeline.release()
    }
}

private actor ControlledPage: OCRPageExecuting {
    nonisolated let entered: AsyncStream<Void>
    private let continuation: AsyncStream<Void>.Continuation
    private let document: OCRDocument
    private var ready = false
    private var pending: CheckedContinuation<OCRDocument, Never>?

    init(document: OCRDocument) {
        self.document = document
        (entered, continuation) = AsyncStream<Void>.makeStream()
    }

    func read(_ input: OCRInput) async throws -> OCRDocument {
        if ready { return document }
        continuation.yield(())
        return await withCheckedContinuation { pending = $0 }
    }

    func complete() {
        ready = true
        let waiter = pending
        pending = nil
        waiter?.resume(returning: document)
        continuation.finish()
    }
}

private actor ControlledLoader {
    nonisolated let entered: AsyncStream<Void>
    private let continuation: AsyncStream<Void>.Continuation
    private let model: any OCRPageExecuting
    private var ready = false
    private var pending: CheckedContinuation<any OCRPageExecuting, Never>?

    init(model: any OCRPageExecuting) {
        self.model = model
        (entered, continuation) = AsyncStream<Void>.makeStream()
    }

    func load() async -> any OCRPageExecuting {
        if ready { return model }
        continuation.yield(())
        return await withCheckedContinuation { pending = $0 }
    }

    func complete() {
        ready = true
        let waiter = pending
        pending = nil
        waiter?.resume(returning: model)
        continuation.finish()
    }
}

func pageFixture() throws -> (input: OCRInput, document: OCRDocument) {
    let image = try OCRImage(width: 28, height: 28, rgb: [UInt8](repeating: 0, count: 28 * 28 * 3))
    let input = OCRInput(name: "page.png", image: image, preview: Data(), sha256: String(repeating: "a", count: 64))
    let timing = OCRDocument.Seconds(vision: 0.1, firstToken: 0.2, total: 1)
    let tokens = OCRDocument.Tokens(prompt: 4, image: 1, output: 2)
    let layout = OCRDocument.Layout(
        parsed: false, text: "", skippedBoxes: [], seconds: timing, tokens: tokens, finishedBy: "eos")
    let block = OCRDocument.Block(
        index: 0, layout: nil, box: OCRBox(x0: 0, y0: 0, x1: 28, y1: 28), text: "Read text",
        seconds: timing, tokens: tokens, finishedBy: "eos")
    return (
        input, OCRDocument(bundle: "fixture", input: input, layout: layout, blocks: [block], generateS: 1, loadS: 0.5)
    )
}
