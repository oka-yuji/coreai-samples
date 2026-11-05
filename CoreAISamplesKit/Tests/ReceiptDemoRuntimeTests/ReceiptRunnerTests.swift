import FeatureStore
import Foundation
import ReceiptCore
import ReceiptFeature
import Testing

@testable import ReceiptDemoRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ReceiptRunnerTests {
    @Test("A cancelled read is joined and its committed result is returned by stop")
    func reconcilesRead() async throws {
        let service = ReceiptServiceFake()
        let runner = ReceiptRunner(service: service)
        let ticket = ReceiptTicket(session: 1, request: 1, item: 1)
        var readActions: [ReceiptAction] = []
        let read = Task { await runner.run(.read(ticket, URL(filePath: "/receipt.png"))) { readActions.append($0) } }
        await service.started.wait()
        read.cancel()
        let stopTicket = ReceiptTicket(session: 1, request: 2)
        var stopActions: [ReceiptAction] = []
        let stop = Task { await runner.run(.stop(stopTicket, interrupted: ticket)) { stopActions.append($0) } }
        await service.stopStarted.wait()
        #expect(stopActions.isEmpty)
        service.readGate.open()
        await read.value
        await stop.value
        #expect(readActions.isEmpty)
        #expect(
            stopActions == [
                .stopped(stopTicket, ReceiptFinishedOperation(ticket: ticket, completion: .read(service.readResult)))
            ])
    }

    @Test("Release waits for a service that ignores cancellation and a revoked owner cannot start work")
    func releaseAndPermission() async throws {
        let service = ReceiptServiceFake()
        let permission = ReceiptTestPermission()
        let runner = ReceiptRunner(service: service, permits: { permission.active })
        let ticket = ReceiptTicket(session: 1, request: 1, item: 1)
        let read = Task { await runner.run(.read(ticket, URL(filePath: "/receipt.png")), send: { _ in }) }
        await service.started.wait()
        permission.active = false
        var released = false
        let release = Task {
            await runner.release()
            released = true
        }
        await service.releaseStarted.wait()
        #expect(!released)
        service.readGate.open()
        await read.value
        await release.value
        #expect(released)
        await runner.run(
            .read(ticket, URL(filePath: "/receipt.png")), send: { _ in Issue.record("A revoked runner sent an action") }
        )
        #expect(service.readCalls == 1)
    }
}

@MainActor
final class ReceiptGate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func wait() async { if !isOpen { await withCheckedContinuation { waiters.append($0) } } }
    func open() {
        isOpen = true
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume() }
    }
}

@MainActor
final class ReceiptServiceFake: ReceiptService {
    let started = ReceiptGate()
    let readGate = ReceiptGate()
    let stopStarted = ReceiptGate()
    let releaseStarted = ReceiptGate()
    var readCalls = 0
    var releaseCalls = 0
    var releaseGate: ReceiptGate?
    var exportGate: ReceiptGate?
    let exportStarted = ReceiptGate()
    let readResult = ReceiptRead(
        record: ReceiptRecord(
            id: 1, sourceName: "receipt.png", sourceExtension: "png", sourceSHA256: String(repeating: "a", count: 64),
            text: "Shop",
            extraction: ReceiptExtraction(fields: ReceiptFields(merchant: "Shop", date: "2026-11-06", amount: "1000"))),
        previews: [Data([1])], seconds: 0.1)

    func samples() async throws -> [ReceiptInput] { [] }
    func prepare(bundle: URL) async throws {}
    func read(id: Int, url: URL) async throws -> ReceiptRead {
        readCalls += 1
        started.open()
        await readGate.wait()
        return readResult
    }
    func classify(_ record: ReceiptRecord) async throws -> ReceiptDecision {
        ReceiptDecision(
            classification: try ReceiptClassification(probabilities: [0, 0, 1, 0, 0, 0, 0, 0]), tokenCount: 100,
            seconds: 0.1)
    }
    func remove(id: Int) async throws {}
    func export(_ ledger: ReceiptLedger, to directory: URL) async throws -> ReceiptSaved {
        exportStarted.open()
        if let exportGate { await exportGate.wait() }
        return ReceiptSaved(directory: directory, count: ledger.rows.count, totalYen: ledger.totalYen)
    }
    func stop() async { stopStarted.open() }
    func release() async {
        releaseCalls += 1
        releaseStarted.open()
        if let releaseGate { await releaseGate.wait() }
    }
}

@MainActor
final class ReceiptTestPermission { var active = true }
