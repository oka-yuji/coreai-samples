import FeatureStore
import Foundation
import ModelLibraryFeature
import ReceiptCore
import ReceiptFeature
import Testing

@testable import ReceiptDemoRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ReceiptSessionTests {
    @Test("Owner replacement waits for release and the ended store cannot restart")
    func owners() async throws {
        var services: [ReceiptServiceFake] = []
        let sessions = ReceiptDemoSessions(makeService: {
            let service = ReceiptServiceFake()
            services.append(service)
            return service
        })
        let first = session(owner: UUID())
        let second = session(owner: UUID())
        let binding = sessions.bind(first, permits: { true })
        let old = try #require(binding)
        #expect(sessions.bind(first, permits: { true }) === old)
        #expect(sessions.bind(second, permits: { true }) == nil)
        let gate = ReceiptGate()
        services[0].releaseGate = gate
        let stopping = Task { await sessions.stop(first) }
        await services[0].releaseStarted.wait()
        #expect(old.state.ended && sessions.bind(second, permits: { true }) == nil)
        gate.open()
        await stopping.value
        let currentBinding = sessions.bind(second, permits: { true })
        let current = try #require(currentBinding)
        #expect(current !== old && current.state.items.isEmpty && services.count == 2)
        old.send(.appeared)
        old.send(.add([ReceiptInput(url: URL(filePath: "/receipt.png"))]))
        #expect(old.state.ended && old.state.items.isEmpty && services[0].readCalls == 0)
        await sessions.stop(first)
        #expect(current.state.presented && !current.state.ended)
        await sessions.stop(second)
        #expect(services[1].releaseCalls > 0)
        let third = session(owner: UUID())
        weak let unpresented = sessions.bind(third, permits: { true })
        #expect(unpresented != nil)
        await sessions.stop(third)
        #expect(unpresented == nil)
    }

    @Test("The real Store reconciles a cancelled read and continues without importing it twice")
    func storeRead() async throws {
        let service = ReceiptServiceFake()
        let runner = ReceiptRunner(service: service)
        let events = AsyncStream<ReceiptAction>.makeStream()
        let store = Store<ReceiptFeature>(initialState: ReceiptState(bundle: URL(filePath: "/model"))) { effect, send in
            await runner.run(effect) { action in
                send(action)
                events.continuation.yield(action)
            }
        }
        store.send(.appeared)
        store.send(.add([ReceiptInput(url: URL(filePath: "/receipt.png"))]))
        await service.started.wait()
        store.send(.stop)
        await service.stopStarted.wait()
        #expect(store.state.phase == .stopping)
        service.readGate.open()
        var received = events.stream.makeAsyncIterator()
        while let action = await received.next() { if case .stopped = action { break } }
        #expect(store.state.readCount == 1 && store.state.phase == .idle)
        store.send(.process)
        while let action = await received.next() {
            if case .finished = action, store.state.phase == .idle { break }
        }
        #expect(service.readCalls == 1 && store.state.items[0].record?.fields.category == .supplies)
        store.send(.ended)
        await runner.release()
        events.continuation.finish()
    }

    @Test("The real Store reports an export that committed while cancellation was pending")
    func storeExport() async throws {
        let service = ReceiptServiceFake()
        service.readGate.open()
        let exportGate = ReceiptGate()
        service.exportGate = exportGate
        let runner = ReceiptRunner(service: service)
        let events = AsyncStream<ReceiptAction>.makeStream()
        let store = Store<ReceiptFeature>(initialState: ReceiptState(bundle: URL(filePath: "/model"))) { effect, send in
            await runner.run(effect) { action in
                send(action)
                events.continuation.yield(action)
            }
        }
        store.send(.appeared)
        store.send(.add([ReceiptInput(url: URL(filePath: "/receipt.png"))]))
        var received = events.stream.makeAsyncIterator()
        while let action = await received.next() { if case .finished = action, store.state.phase == .idle { break } }
        store.send(.confirm(1))
        store.send(.requestExport)
        let panel = try #require(store.state.exportSelection)
        store.send(.exportSelected(panel.id, URL(filePath: "/output")))
        await service.exportStarted.wait()
        store.send(.stop)
        await service.stopStarted.wait()
        exportGate.open()
        while let action = await received.next() { if case .stopped = action { break } }
        #expect(store.state.saved?.totalYen == 1000 && store.state.saved?.count == 1)
        store.send(.ended)
        await runner.release()
        events.continuation.finish()
    }

    private func session(owner: UUID) -> ModelDemoSession {
        ModelDemoSession(
            id: LibraryTicket(owner: owner, number: 1), selection: ReceiptDemoCatalog.selection,
            resource: ValidatedModelResource(
                modelID: "jev-27b", artifactID: "jev-27b-coreai-uint4", contract: "jev-27b-coreai-uint4-v1",
                revision: String(repeating: "a", count: 40), upstreamRevision: nil,
                directory: URL(filePath: "/fixture/model"), sizeBytes: 0, sourceKind: .huggingFace))
    }
}
