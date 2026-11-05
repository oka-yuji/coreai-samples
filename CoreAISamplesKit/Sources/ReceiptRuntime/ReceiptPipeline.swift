import Foundation
import ReceiptCore

public actor ReceiptPipeline {
    typealias Reader = @Sendable (Data, String) async throws -> ReceiptRecognizedDocument
    typealias Loader = @Sendable (URL) async throws -> any ReceiptDecisionExecuting
    typealias ExportCheckpoint = @Sendable (Int) async throws -> Void

    private let reader: Reader
    private let loader: Loader
    private let exportCheckpoint: ExportCheckpoint
    private let onReleaseWaiting: @Sendable () -> Void
    private var parser: ReceiptTextParser?
    private var model: (any ReceiptDecisionExecuting)?
    private var modelURL: URL?
    private var sources: [Int: ReceiptSourceCopy] = [:]
    private var busy = false
    private var released = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init() {
        let vision = ReceiptVisionReader()
        reader = { try await vision.read($0, fileExtension: $1) }
        loader = CoreAIReceiptDecision.load
        exportCheckpoint = { _ in }
        onReleaseWaiting = {}
    }

    init(
        rules: ReceiptExtractionRules, reader: @escaping Reader, loader: @escaping Loader,
        exportCheckpoint: @escaping ExportCheckpoint = { _ in }, onReleaseWaiting: @escaping @Sendable () -> Void = {}
    ) throws {
        parser = try ReceiptTextParser(rules: rules)
        self.reader = reader
        self.loader = loader
        self.exportCheckpoint = exportCheckpoint
        self.onReleaseWaiting = onReleaseWaiting
    }

    public func prepare(bundle: URL) async throws {
        try begin()
        defer { finish() }
        guard bundle.isFileURL else { throw ReceiptRuntimeError.invalidSource }
        let url = bundle.standardizedFileURL
        if model != nil {
            guard modelURL == url else { throw ReceiptRuntimeError.busy }
            return
        }
        let loaded = try await loader(url)
        try checkActive()
        model = loaded
        modelURL = url
    }

    public func read(id: Int, url: URL) async throws -> ReceiptReadResult {
        try begin()
        defer { finish() }
        guard id > 0, sources[id] == nil else { throw ReceiptValidationError.duplicateIdentifier }
        guard sources.count < 20 else { throw ReceiptRuntimeError.tooManyReceipts }
        let start = ContinuousClock.now
        let source = try ReceiptFileReader.read(url)
        guard !sources.values.contains(where: { $0.sha256 == source.sha256 }) else {
            throw ReceiptRuntimeError.duplicateReceipt
        }
        let recognized: ReceiptRecognizedDocument
        do { recognized = try await reader(source.bytes, source.fileExtension) } catch {
            try Task.checkCancellation()
            throw error
        }
        try checkActive()
        guard ["png", "jpg", "jpeg", "heic", "heif", "tif", "tiff", "pdf"].contains(recognized.fileExtension),
            (1...5).contains(recognized.previews.count),
            recognized.unreadablePages.allSatisfy({ (1...recognized.previews.count).contains($0) })
        else { throw ReceiptRuntimeError.unsupportedFile }
        if parser == nil { parser = try ReceiptTextParser(rules: Self.extractionRules()) }
        guard let parser else { throw ReceiptValidationError.invalidRules }
        var extraction = parser.extract(recognized.lines)
        if !recognized.unreadablePages.isEmpty { extraction.issues.append(.unreadablePage) }
        let record = ReceiptRecord(
            id: id, sourceName: source.name, sourceExtension: recognized.fileExtension, sourceSHA256: source.sha256,
            text: recognized.lines.map(\.text).joined(separator: "\n"), extraction: extraction)
        sources[id] = ReceiptSourceCopy(
            name: source.name, fileExtension: recognized.fileExtension, bytes: source.bytes, sha256: source.sha256,
            text: record.text)
        return ReceiptReadResult(
            record: record, previews: recognized.previews, seconds: start.duration(to: .now) / .seconds(1))
    }

    public func classify(_ record: ReceiptRecord) async throws -> ReceiptDecisionResult {
        try begin()
        defer { finish() }
        try validateSource(record.id, name: record.sourceName, sha256: record.sourceSHA256)
        guard sources[record.id]?.text == record.text else { throw ReceiptRuntimeError.sourceMismatch }
        guard !record.issues.contains(.unreadablePage),
            !record.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw ReceiptRuntimeError.incompleteText
        }
        guard let model else { throw ReceiptRuntimeError.noModel }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode([
            "receipt_text": record.text, "merchant": record.fields.merchant, "purchase_purpose": record.fields.purpose,
        ])
        let input = ReceiptDecisionInput(
            state: String(decoding: data, as: UTF8.self),
            question: ReceiptCategory.classificationQuestion,
            options: ReceiptCategory.allCases.map(\.decisionOption))
        let start = ContinuousClock.now
        let answer = try await model.decide(input)
        try checkActive()
        guard answer.tokenCount > 0 else { throw ReceiptValidationError.invalidClassification }
        return ReceiptDecisionResult(
            classification: try ReceiptClassification(probabilities: answer.probabilities),
            tokenCount: answer.tokenCount,
            seconds: start.duration(to: .now) / .seconds(1))
    }

    public func export(_ ledger: ReceiptLedger, to directory: URL, identifier: UUID = UUID()) async throws
        -> ReceiptExportResult
    {
        try begin()
        defer { finish() }
        for row in ledger.rows {
            try validateSource(row.id, name: row.sourceName, sha256: row.sourceSHA256)
            guard sources[row.id]?.fileExtension == URL(filePath: row.relativePath).pathExtension else {
                throw ReceiptRuntimeError.sourceMismatch
            }
        }
        let accessed = directory.startAccessingSecurityScopedResource()
        defer { if accessed { directory.stopAccessingSecurityScopedResource() } }
        let transaction = try ReceiptExportTransaction(directory: directory, identifier: identifier)
        do {
            for (index, row) in ledger.rows.enumerated() {
                try await exportCheckpoint(index)
                try checkActive()
                guard let source = sources[row.id], ReceiptFileReader.digest(source.bytes) == row.sourceSHA256 else {
                    throw ReceiptRuntimeError.sourceMismatch
                }
                try transaction.write(source.bytes, path: row.relativePath)
            }
            try transaction.write(ledger.csv, path: "ledger.csv")
            try transaction.writeLedger(ledger)
            try await exportCheckpoint(ledger.rows.count)
            try checkActive()
            let destination = try transaction.publish()
            return ReceiptExportResult(directory: destination, count: ledger.rows.count, totalYen: ledger.totalYen)
        } catch {
            do { try transaction.cleanUp() } catch { throw ReceiptRuntimeError.cleanupFailed }
            throw error
        }
    }

    public func remove(id: Int) throws {
        try begin()
        defer { finish() }
        sources[id] = nil
    }

    public func waitForIdle() async {
        if busy { await withCheckedContinuation { waiters.append($0) } }
    }

    public func release() async {
        released = true
        if busy {
            onReleaseWaiting()
            await withCheckedContinuation { waiters.append($0) }
        } else {
            discard()
        }
    }

    private func begin() throws {
        try checkActive()
        guard !busy else { throw ReceiptRuntimeError.busy }
        busy = true
    }

    private func checkActive() throws {
        try Task.checkCancellation()
        guard !released else { throw ReceiptRuntimeError.released }
    }

    private func finish() {
        busy = false
        if released { discard() }
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume() }
    }

    private func discard() {
        model = nil
        modelURL = nil
        sources.removeAll()
    }

    private func validateSource(_ id: Int, name: String, sha256: String) throws {
        guard let source = sources[id], source.name == name, source.sha256 == sha256 else {
            throw ReceiptRuntimeError.sourceMismatch
        }
    }

    static func extractionRules() throws -> ReceiptExtractionRules {
        guard
            let url = Bundle.module.url(forResource: "receipt-rules", withExtension: "json", subdirectory: "Resources")
        else {
            throw ReceiptValidationError.invalidRules
        }
        return try JSONDecoder().decode(ReceiptExtractionRules.self, from: Data(contentsOf: url))
    }
}
