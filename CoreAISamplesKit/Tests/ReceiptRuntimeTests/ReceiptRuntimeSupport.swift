import Foundation
import ReceiptCore

@testable import ReceiptRuntime

struct ReceiptTestFiles {
    let root: URL

    init() throws {
        root = FileManager.default.temporaryDirectory.appending(path: "ReceiptTests-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    }

    func file(_ name: String = "receipt.png", bytes: Data = Data("original receipt".utf8)) throws -> URL {
        let url = root.appending(path: name)
        try bytes.write(to: url, options: .withoutOverwriting)
        return url
    }

    func folder(_ name: String = "export") throws -> URL {
        let url = root.appending(path: name)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }

    func cleanUp() { try? FileManager.default.removeItem(at: root) }
}

enum ReceiptTestFailure: Error { case injected }

actor ReceiptTestReader {
    private var failures = 1

    func read(_ bytes: Data, _ ext: String) async throws -> ReceiptRecognizedDocument {
        if failures > 0 {
            failures -= 1
            throw ReceiptTestFailure.injected
        }
        return try await receiptRecognition(bytes, ext)
    }
}

actor ReceiptTestLoader {
    let engine: ReceiptTestEngine
    let gate: ReceiptTestGate?
    private var failures: Int
    private(set) var calls = 0

    init(engine: ReceiptTestEngine, failures: Int = 0, gate: ReceiptTestGate? = nil) {
        self.engine = engine
        self.failures = failures
        self.gate = gate
    }

    func load(_ url: URL) async throws -> any ReceiptDecisionExecuting {
        calls += 1
        if let gate { await gate.block() }
        if failures > 0 {
            failures -= 1
            throw ReceiptTestFailure.injected
        }
        return engine
    }
}

actor ReceiptTestGate {
    private var started = false
    private var opened = false
    private var waiting: [CheckedContinuation<Void, Never>] = []
    private var startWaiters: [CheckedContinuation<Void, Never>] = []

    func block() async {
        started = true
        let starters = startWaiters
        startWaiters.removeAll()
        for waiter in starters { waiter.resume() }
        if !opened { await withCheckedContinuation { waiting.append($0) } }
    }

    func waitForStart() async {
        if !started { await withCheckedContinuation { startWaiters.append($0) } }
    }

    func open() {
        opened = true
        let pending = waiting
        waiting.removeAll()
        for waiter in pending { waiter.resume() }
    }
}

actor ReceiptTestEngine: ReceiptDecisionExecuting {
    let gate: ReceiptTestGate?
    var values: ReceiptDecisionValues
    var received: [ReceiptDecisionInput] = []
    var failures = 0

    init(gate: ReceiptTestGate? = nil, probabilities: [Float] = [0, 0, 1, 0, 0, 0, 0, 0]) {
        self.gate = gate
        values = ReceiptDecisionValues(probabilities: probabilities, tokenCount: 120)
    }

    func decide(_ input: ReceiptDecisionInput) async throws -> ReceiptDecisionValues {
        received.append(input)
        if let gate { await gate.block() }
        if failures > 0 {
            failures -= 1
            throw ReceiptTestFailure.injected
        }
        return values
    }

    func failOnce() { failures = 1 }
    func set(_ probabilities: [Float]) { values = ReceiptDecisionValues(probabilities: probabilities, tokenCount: 120) }
}

func receiptRecognition(_ bytes: Data, _ ext: String) async throws -> ReceiptRecognizedDocument {
    ReceiptRecognizedDocument(
        lines: [ReceiptTextLine("Sample Shop\n2026-11-06\nPaper and pens\nTotal JPY 1000")],
        previews: [Data("preview".utf8)], fileExtension: ext)
}

func receiptPipeline(
    engine: ReceiptTestEngine = ReceiptTestEngine(), reader: @escaping ReceiptPipeline.Reader = receiptRecognition,
    checkpoint: @escaping ReceiptPipeline.ExportCheckpoint = { _ in },
    releaseWaiting: @escaping @Sendable () -> Void = {}
) throws -> ReceiptPipeline {
    try ReceiptPipeline(
        rules: ReceiptPipeline.extractionRules(), reader: reader, loader: { _ in engine },
        exportCheckpoint: checkpoint, onReleaseWaiting: releaseWaiting)
}
