import Darwin
import Foundation
import ReceiptCore
import Testing

@testable import ReceiptRuntime

@Suite(.timeLimit(.minutes(1)))
struct ReceiptPipelineTests {
    @Test("A failed read does not reserve its identifier and a subsequent import can succeed")
    func readRetry() async throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let reader = ReceiptTestReader()
        let pipeline = try receiptPipeline(reader: { try await reader.read($0, $1) })
        let source = try files.file()
        await #expect(throws: ReceiptTestFailure.self) { try await pipeline.read(id: 1, url: source) }
        #expect(try await pipeline.read(id: 1, url: source).record.fields.amount == "1000")
        await pipeline.release()
    }

    @Test("Failed preparation and cancelled preparation both allow a later retry")
    func prepareRetry() async throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let engine = ReceiptTestEngine()
        let loader = ReceiptTestLoader(engine: engine, failures: 1)
        let rules = try ReceiptPipeline.extractionRules()
        let pipeline = try ReceiptPipeline(
            rules: rules, reader: receiptRecognition, loader: { try await loader.load($0) })
        await #expect(throws: ReceiptTestFailure.self) { try await pipeline.prepare(bundle: files.root) }
        try await pipeline.prepare(bundle: files.root)
        try await pipeline.prepare(bundle: files.root)
        #expect(await loader.calls == 2)
        await pipeline.release()
        let gate = ReceiptTestGate()
        let slow = ReceiptTestLoader(engine: engine, gate: gate)
        let retry = try ReceiptPipeline(rules: rules, reader: receiptRecognition, loader: { try await slow.load($0) })
        let operation = Task { try await retry.prepare(bundle: files.root) }
        await gate.waitForStart()
        operation.cancel()
        await #expect(throws: ReceiptRuntimeError.busy) { try await retry.prepare(bundle: files.root) }
        await gate.open()
        await #expect(throws: CancellationError.self) { try await operation.value }
        try await retry.prepare(bundle: files.root)
        #expect(await slow.calls == 2)
        await retry.release()
    }

    @Test("A changed source is rejected before it becomes an imported snapshot")
    func changedSource() throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let source = try files.file()
        #expect(throws: ReceiptRuntimeError.sourceChanged) {
            try ReceiptFileReader.read(source) { try Data("changed".utf8).write(to: source) }
        }
    }

    @Test("The session limit and removal keep retained receipt data bounded")
    func sessionLimit() async throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let pipeline = try receiptPipeline()
        for id in 1...20 {
            _ = try await pipeline.read(id: id, url: files.file("\(id).png", bytes: Data("\(id)".utf8)))
        }
        let extra = try files.file("extra.png", bytes: Data("extra".utf8))
        await #expect(throws: ReceiptRuntimeError.tooManyReceipts) { try await pipeline.read(id: 21, url: extra) }
        try await pipeline.remove(id: 1)
        #expect(try await pipeline.read(id: 21, url: extra).record.id == 21)
        await pipeline.release()
    }
    @Test("Import retains original bytes and rejects duplicate receipts and invalid sources")
    func imports() async throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let source = try files.file()
        let pipeline = try receiptPipeline()
        let result = try await pipeline.read(id: 1, url: source)
        #expect(result.record.fields.amount == "1000")
        #expect(result.record.sourceSHA256 == ReceiptFileReader.digest(Data("original receipt".utf8)))
        #expect(result.previews.count == 1)
        await #expect(throws: ReceiptRuntimeError.duplicateReceipt) { try await pipeline.read(id: 2, url: source) }
        await #expect(throws: ReceiptValidationError.duplicateIdentifier) {
            try await pipeline.read(id: 1, url: source)
        }
        let link = files.root.appending(path: "link.png")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: source)
        await #expect(throws: ReceiptRuntimeError.invalidSource) { try await pipeline.read(id: 3, url: link) }
        let large = try files.file("large.png", bytes: Data())
        let handle = try FileHandle(forWritingTo: large)
        try handle.truncate(atOffset: UInt64(ReceiptFileReader.maximumBytes + 1))
        try handle.close()
        await #expect(throws: ReceiptRuntimeError.tooLarge) { try await pipeline.read(id: 4, url: large) }
        let pipe = files.root.appending(path: "pipe.png")
        #expect(mkfifo(pipe.path, 0o600) == 0)
        await #expect(throws: ReceiptRuntimeError.invalidSource) { try await pipeline.read(id: 5, url: pipe) }
        #expect(try Data(contentsOf: source) == Data("original receipt".utf8))
        await pipeline.release()
    }

    @Test("Classification uses the fixed choices and corrected purpose without changing receipt fields")
    func classification() async throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let engine = ReceiptTestEngine()
        let pipeline = try receiptPipeline(engine: engine)
        let source = try files.file()
        var record = try await pipeline.read(id: 1, url: source).record
        record.edit(.purpose, value: "Office supplies for the project")
        let before = record.fields
        await #expect(throws: ReceiptRuntimeError.noModel) { try await pipeline.classify(record) }
        try await pipeline.prepare(bundle: files.root)
        let result = try await pipeline.classify(record)
        #expect(result.classification.category == .supplies)
        #expect(result.tokenCount == 120)
        #expect(record.fields == before)
        let received = await engine.received
        let input = try #require(received.first)
        #expect(input.options == ReceiptCategory.allCases.map(\.decisionOption))
        let state = try JSONDecoder().decode([String: String].self, from: Data(input.state.utf8))
        #expect(state["purchase_purpose"] == record.fields.purpose)
        #expect(state["receipt_text"] == record.text)
        await engine.set([0.4, 0.3, 0.3, 0, 0, 0, 0, 0])
        #expect(try await pipeline.classify(record).classification.category == .review)
        await engine.set([Float.nan, 0, 1, 0, 0, 0, 0, 0])
        await #expect(throws: ReceiptValidationError.invalidClassification) { try await pipeline.classify(record) }
        await engine.set([0, 0, 1, 0, 0, 0, 0, 0])
        await engine.failOnce()
        await #expect(throws: ReceiptTestFailure.self) { try await pipeline.classify(record) }
        #expect(try await pipeline.classify(record).classification.category == .supplies)
        await pipeline.release()
    }

    @Test("Cancellation waits for an uncooperative decision before the pipeline can run again")
    func cancellation() async throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let gate = ReceiptTestGate()
        let pipeline = try receiptPipeline(engine: ReceiptTestEngine(gate: gate))
        let record = try await pipeline.read(id: 1, url: files.file()).record
        try await pipeline.prepare(bundle: files.root)
        let operation = Task { try await pipeline.classify(record) }
        await gate.waitForStart()
        operation.cancel()
        await #expect(throws: ReceiptRuntimeError.busy) { try await pipeline.classify(record) }
        let stopped = Task { await pipeline.waitForIdle() }
        await gate.open()
        await #expect(throws: CancellationError.self) { try await operation.value }
        await stopped.value
        #expect(try await pipeline.classify(record).classification.category == .supplies)
        await pipeline.release()
    }

    @Test("All release waiters join the active operation and the released pipeline cannot be reused")
    func release() async throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let gate = ReceiptTestGate()
        let waiting = AsyncStream<Void>.makeStream()
        let pipeline = try receiptPipeline(
            engine: ReceiptTestEngine(gate: gate), releaseWaiting: { waiting.continuation.yield(()) })
        let record = try await pipeline.read(id: 1, url: files.file()).record
        try await pipeline.prepare(bundle: files.root)
        let operation = Task { try await pipeline.classify(record) }
        await gate.waitForStart()
        let first = Task { await pipeline.release() }
        let second = Task { await pipeline.release() }
        var iterator = waiting.stream.makeAsyncIterator()
        _ = await iterator.next()
        _ = await iterator.next()
        await #expect(throws: ReceiptRuntimeError.released) { try await pipeline.prepare(bundle: files.root) }
        await gate.open()
        await #expect(throws: ReceiptRuntimeError.released) { try await operation.value }
        await first.value
        await second.value
        waiting.continuation.finish()
        await #expect(throws: ReceiptRuntimeError.released) {
            try await pipeline.read(id: 2, url: files.file("later.png"))
        }
    }

    @Test("Incomplete recognition stays reviewable and never produces a model classification")
    func unreadablePage() async throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let engine = ReceiptTestEngine()
        let pipeline = try receiptPipeline(engine: engine) { _, ext in
            ReceiptRecognizedDocument(
                lines: [ReceiptTextLine("Shop\n2026-11-06\nTotal JPY 1000")], previews: [Data()],
                fileExtension: ext, unreadablePages: [1])
        }
        var record = try await pipeline.read(id: 1, url: files.file()).record
        try await pipeline.prepare(bundle: files.root)
        await #expect(throws: ReceiptRuntimeError.incompleteText) { try await pipeline.classify(record) }
        #expect(await engine.received.isEmpty)
        record.setCategory(.supplies)
        #expect(!record.canConfirmTogether)
        #expect(record.canConfirm)
        await pipeline.release()
    }
}
