import Foundation
import ReceiptCore
import Testing

@testable import ReceiptRuntime

@Suite(.timeLimit(.minutes(1)))
struct ReceiptExportTests {
    @Test("Export publishes a complete ledger with the imported snapshot even if the input file changes")
    func snapshot() async throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let original = Data("first snapshot".utf8)
        let input = try files.file(bytes: original)
        let pipeline = try receiptPipeline()
        var record = try await pipeline.read(id: 1, url: input).record
        record.setCategory(.supplies)
        try record.confirm()
        let ledger = try ReceiptLedger(records: [record])
        try Data("changed after import".utf8).write(to: input)
        let output = try files.folder()
        let result = try await pipeline.export(ledger, to: output)
        #expect(result.count == 1)
        #expect(result.totalYen == 1000)
        #expect(try Data(contentsOf: result.directory.appending(path: ledger.rows[0].relativePath)) == original)
        #expect(try Data(contentsOf: result.directory.appending(path: "ledger.csv")) == ledger.csv)
        let json = try Data(contentsOf: result.directory.appending(path: "ledger.json"))
        #expect(String(decoding: json, as: UTF8.self).contains(record.sourceSHA256))
        #expect(try Data(contentsOf: input) == Data("changed after import".utf8))
        #expect(try FileManager.default.contentsOfDirectory(atPath: output.path).count == 1)
        await pipeline.release()
    }

    @Test("A stale fingerprint and an existing destination cannot publish or overwrite receipt files")
    func refusesReplacement() async throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let pipeline = try receiptPipeline()
        let input = try files.file()
        var record = try await pipeline.read(id: 1, url: input).record
        record.setCategory(.supplies)
        try record.confirm()
        var forged = ReceiptRecord(
            id: 1, sourceName: record.sourceName, sourceExtension: record.sourceExtension,
            sourceSHA256: String(repeating: "b", count: 64), text: record.text,
            extraction: ReceiptExtraction(fields: record.fields))
        try forged.confirm()
        let output = try files.folder()
        await #expect(throws: ReceiptRuntimeError.sourceMismatch) {
            try await pipeline.export(ReceiptLedger(records: [forged]), to: output)
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: output.path).isEmpty)
        let identifier = UUID()
        let existing = output.appending(path: "ReceiptDesk-" + identifier.uuidString)
        try FileManager.default.createDirectory(at: existing, withIntermediateDirectories: false)
        let sentinel = existing.appending(path: "keep.txt")
        try Data("keep".utf8).write(to: sentinel)
        await #expect(throws: ReceiptRuntimeError.destinationExists) {
            try await pipeline.export(ReceiptLedger(records: [record]), to: output, identifier: identifier)
        }
        #expect(try Data(contentsOf: sentinel) == Data("keep".utf8))
        await pipeline.release()
    }

    @Test("Cancelling after a receipt write removes the unpublished transaction before stop completes")
    func cancellation() async throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let gate = ReceiptTestGate()
        let pipeline = try receiptPipeline(checkpoint: { index in if index == 1 { await gate.block() } })
        var records: [ReceiptRecord] = []
        for id in 1...2 {
            var record = try await pipeline.read(id: id, url: files.file("\(id).png", bytes: Data("\(id)".utf8))).record
            record.setCategory(.supplies)
            try record.confirm()
            records.append(record)
        }
        let ledger = try ReceiptLedger(records: records)
        let output = try files.folder()
        let export = Task { try await pipeline.export(ledger, to: output) }
        await gate.waitForStart()
        #expect(try FileManager.default.contentsOfDirectory(atPath: output.path).count == 1)
        export.cancel()
        let stopped = Task { await pipeline.waitForIdle() }
        await gate.open()
        await #expect(throws: CancellationError.self) { try await export.value }
        await stopped.value
        #expect(try FileManager.default.contentsOfDirectory(atPath: output.path).isEmpty)
        let result = try await pipeline.export(ledger, to: output)
        #expect(result.count == 2)
        await pipeline.release()
    }

    @Test("A write failure leaves no published ledger and preserves the source")
    func writeFailure() async throws {
        let files = try ReceiptTestFiles()
        defer { files.cleanUp() }
        let pipeline = try receiptPipeline(checkpoint: { index in if index == 1 { throw ReceiptTestFailure.injected } })
        let input = try files.file()
        var record = try await pipeline.read(id: 1, url: input).record
        record.setCategory(.supplies)
        try record.confirm()
        let output = try files.folder()
        await #expect(throws: ReceiptTestFailure.self) {
            try await pipeline.export(ReceiptLedger(records: [record]), to: output)
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: output.path).isEmpty)
        #expect(try Data(contentsOf: input) == Data("original receipt".utf8))
        await pipeline.release()
    }
}
