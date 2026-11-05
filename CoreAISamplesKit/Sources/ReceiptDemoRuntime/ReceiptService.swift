import Foundation
import ReceiptCore
import ReceiptFeature
import ReceiptRuntime

@MainActor
public protocol ReceiptService {
    func samples() async throws -> [ReceiptInput]
    func prepare(bundle: URL) async throws
    func read(id: Int, url: URL) async throws -> ReceiptRead
    func classify(_ record: ReceiptRecord) async throws -> ReceiptDecision
    func remove(id: Int) async throws
    func export(_ ledger: ReceiptLedger, to directory: URL) async throws -> ReceiptSaved
    func stop() async
    func release() async
}

@MainActor
public final class LiveReceiptService: ReceiptService {
    private let pipeline: ReceiptPipeline

    public init() { pipeline = ReceiptPipeline() }

    public func samples() async throws -> [ReceiptInput] {
        try await ReceiptPipeline.samples().map { ReceiptInput(url: $0.url, purpose: $0.purpose, title: $0.title) }
    }

    public func prepare(bundle: URL) async throws { try await pipeline.prepare(bundle: bundle) }

    public func read(id: Int, url: URL) async throws -> ReceiptRead {
        let result = try await pipeline.read(id: id, url: url)
        return ReceiptRead(record: result.record, previews: result.previews, seconds: result.seconds)
    }

    public func classify(_ record: ReceiptRecord) async throws -> ReceiptDecision {
        let result = try await pipeline.classify(record)
        return ReceiptDecision(
            classification: result.classification, tokenCount: result.tokenCount, seconds: result.seconds)
    }

    public func remove(id: Int) async throws { try await pipeline.remove(id: id) }

    public func export(_ ledger: ReceiptLedger, to directory: URL) async throws -> ReceiptSaved {
        let result = try await pipeline.export(ledger, to: directory)
        return ReceiptSaved(directory: result.directory, count: result.count, totalYen: result.totalYen)
    }

    public func stop() async { await pipeline.waitForIdle() }
    public func release() async { await pipeline.release() }
}
