import FeatureStore
import Foundation
import ReceiptCore
import ReceiptDemoRuntime
import ReceiptFeature
import Testing

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ReceiptOperationBoundaryTests {
    @Test("A stop before the operation task starts does not strand the queue")
    func stopBeforeStart() async throws {
        let service = ReceiptBoundaryService()
        let (store, runner, stream) = desk(service)
        store.send(.add([ReceiptInput(url: URL(filePath: "/receipt.png"))]))
        store.send(.stop)
        var events = stream.stream.makeAsyncIterator()
        while let action = await events.next() { if case .stopped = action { break } }
        #expect(store.state.phase == .idle)
        #expect(service.prepareCalls == 0 && service.readCalls == 0)
        store.send(.process)
        while let action = await events.next() { if case .finished = action, store.state.phase == .idle { break } }
        #expect(service.prepareCalls == 1 && service.readCalls == 1)
        #expect(store.state.items[0].record?.fields.category == .supplies)
        store.send(.ended)
        await runner.release()
        stream.continuation.finish()
    }

    @Test("A cancelled remove commit is reflected before the same input is added again")
    func committedRemove() async throws {
        let service = ReceiptBoundaryService()
        let removal = ReceiptGate()
        service.removeGate = removal
        let (store, runner, stream) = desk(service)
        store.send(.add([ReceiptInput(url: URL(filePath: "/receipt.png"))]))
        var events = stream.stream.makeAsyncIterator()
        while let action = await events.next() { if case .finished = action, store.state.phase == .idle { break } }
        store.send(.remove(1))
        await service.removeStarted.wait()
        store.send(.stop)
        await service.stopStarted.wait()
        #expect(store.state.items.count == 1 && store.state.phase == .stopping)
        removal.open()
        while let action = await events.next() { if case .stopped = action { break } }
        #expect(store.state.items.isEmpty && service.sources.isEmpty)
        store.send(.add([ReceiptInput(url: URL(filePath: "/receipt.png"))]))
        while let action = await events.next() { if case .finished = action, store.state.phase == .idle { break } }
        #expect(store.state.items.map(\.id) == [2] && service.sources == [2])
        #expect(service.readCalls == 2)
        store.send(.ended)
        await runner.release()
        stream.continuation.finish()
    }

    @Test("A prepared model is acknowledged by stop and is not prepared twice")
    func committedPreparation() async throws {
        let service = ReceiptBoundaryService()
        let preparation = ReceiptGate()
        service.prepareGate = preparation
        let (store, runner, stream) = desk(service)
        store.send(.add([ReceiptInput(url: URL(filePath: "/receipt.png"))]))
        await service.prepareStarted.wait()
        store.send(.stop)
        await service.stopStarted.wait()
        preparation.open()
        var events = stream.stream.makeAsyncIterator()
        while let action = await events.next() { if case .stopped = action { break } }
        #expect(store.state.modelPrepared && service.readCalls == 0)
        store.send(.process)
        while let action = await events.next() { if case .finished = action, store.state.phase == .idle { break } }
        #expect(service.prepareCalls == 1 && service.readCalls == 1)
        store.send(.ended)
        await runner.release()
        stream.continuation.finish()
    }

    @Test("A committed classification stops the rest of the batch and does not classify twice")
    func committedClassification() async throws {
        let service = ReceiptBoundaryService()
        let classification = ReceiptGate()
        service.classifyGate = classification
        let (store, runner, stream) = desk(service)
        store.send(.add([1, 2].map { ReceiptInput(url: URL(filePath: "/receipt\($0).png")) }))
        await service.classifyStarted.wait()
        store.send(.stop)
        await service.stopStarted.wait()
        classification.open()
        var events = stream.stream.makeAsyncIterator()
        while let action = await events.next() { if case .stopped = action { break } }
        #expect(store.state.items[0].record?.fields.category == .supplies)
        #expect(!store.state.items[0].needsClassification && store.state.items[1].record == nil)
        #expect(store.state.batchDone == 1)
        store.send(.process)
        while let action = await events.next() { if case .finished = action, store.state.phase == .idle { break } }
        #expect(service.classifyCalls == 2 && service.readCalls == 2)
        store.send(.ended)
        await runner.release()
        stream.continuation.finish()
    }

    private func desk(_ service: ReceiptBoundaryService) -> (
        Store<ReceiptFeature>, ReceiptRunner,
        (stream: AsyncStream<ReceiptAction>, continuation: AsyncStream<ReceiptAction>.Continuation)
    ) {
        let runner = ReceiptRunner(service: service)
        let events = AsyncStream<ReceiptAction>.makeStream()
        let store = Store<ReceiptFeature>(initialState: ReceiptState(bundle: URL(filePath: "/model"))) { effect, send in
            await runner.run(effect) { action in
                send(action)
                events.continuation.yield(action)
            }
        }
        store.send(.appeared)
        return (store, runner, events)
    }
}

@MainActor
private final class ReceiptBoundaryService: ReceiptService {
    var prepareCalls = 0
    var readCalls = 0
    var classifyCalls = 0
    var sources: Set<Int> = []
    var prepareGate: ReceiptGate?
    var classifyGate: ReceiptGate?
    var removeGate: ReceiptGate?
    let prepareStarted = ReceiptGate()
    let classifyStarted = ReceiptGate()
    let removeStarted = ReceiptGate()
    let stopStarted = ReceiptGate()

    func samples() async throws -> [ReceiptInput] { [] }
    func prepare(bundle: URL) async throws {
        prepareCalls += 1
        prepareStarted.open()
        if let prepareGate { await prepareGate.wait() }
    }
    func read(id: Int, url: URL) async throws -> ReceiptRead {
        readCalls += 1
        #expect(!sources.contains(id))
        sources.insert(id)
        return ReceiptRead(
            record: ReceiptRecord(
                id: id, sourceName: url.lastPathComponent, sourceExtension: "png",
                sourceSHA256: String(repeating: "a", count: 64), text: "Shop",
                extraction: ReceiptExtraction(
                    fields: ReceiptFields(merchant: "Shop", date: "2026-11-06", amount: "1000"))),
            previews: [Data([1])], seconds: 0.1)
    }
    func classify(_ record: ReceiptRecord) async throws -> ReceiptDecision {
        classifyCalls += 1
        classifyStarted.open()
        if let classifyGate { await classifyGate.wait() }
        return ReceiptDecision(
            classification: try ReceiptClassification(probabilities: [0, 0, 1, 0, 0, 0, 0, 0]), tokenCount: 1,
            seconds: 0.1)
    }
    func remove(id: Int) async throws {
        removeStarted.open()
        if let removeGate { await removeGate.wait() }
        sources.remove(id)
    }
    func export(_ ledger: ReceiptLedger, to directory: URL) async throws -> ReceiptSaved {
        ReceiptSaved(directory: directory, count: ledger.rows.count, totalYen: ledger.totalYen)
    }
    func stop() async { stopStarted.open() }
    func release() async {}
}
