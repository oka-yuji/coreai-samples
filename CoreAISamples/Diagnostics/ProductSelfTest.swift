import CryptoKit
import FeatureStore
import Foundation
import ProductCore
import ProductFeature
import ProductRuntime

@MainActor
enum ProductSelfTest {
    private static let addresses = [
        "https://www.apple.com/jp/macbook-air/specs/",
        "https://support.microsoft.com/ja-jp/surface/models/surface-laptop-5-specs-and-features",
    ]

    static func run(directory: String, modelFolder: String?) async {
        do {
            let directory = URL(filePath: directory)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            if let modelFolder {
                try await checkCards(directory: directory, folder: URL(filePath: modelFolder))
            } else {
                let service = LiveProductService()
                for (index, address) in addresses.enumerated() {
                    var draft = ProductDraft()
                    draft.address = address
                    let source = try await service.source(draft: draft, capturedAt: Date())
                    try save(source: source, memo: nil, to: directory.appending(path: "source-\(index).json"))
                    draft.mode = .text
                    draft.title = source.title
                    draft.text = source.text
                    let pasted = try await service.source(draft: draft, capturedAt: source.capturedAt)
                    try require(pasted.text == source.text && pasted.url == nil, "Pasted source changed.")
                }
            }
            FileHandle.standardOutput.write(Data("Product scenarios passed.\n".utf8))
        } catch {
            FileHandle.standardError.write(Data(("Product scenario failed: \(error.localizedDescription)\n").utf8))
            exit(1)
        }
    }

    private static func checkCards(directory: URL, folder: URL) async throws {
        let harness = ProductScenarioHarness(folder: folder)
        let store = harness.store
        defer { store.send(.closed) }
        for (index, address) in addresses.enumerated() {
            var draft = ProductDraft()
            draft.address = address
            store.send(.draftChanged(draft))
            store.send(.submit(Date()))
            await harness.settle()
            try require(store.state.cards.count == index + 1, failure(in: store.state))
            guard let card = store.state.cards.last else { throw ProductError.invalidEvidence }
            try require(!card.memo.evidence.isEmpty, "No specification candidates were found.")
            try save(source: card.memo.source, memo: card.memo, to: directory.appending(path: "card-\(index).json"))
        }
        guard let original = store.state.cards.first?.memo.source else { throw ProductError.invalidEvidence }
        var pasted = ProductDraft(mode: .text)
        pasted.title = original.title
        pasted.text = original.text
        store.send(.draftChanged(pasted))
        store.send(.submit(Date()))
        await harness.settle()
        try require(store.state.cards.count == 3, failure(in: store.state))
        guard let card = store.state.cards.last else { throw ProductError.invalidEvidence }
        try require(card.memo.source.url == nil && card.memo.source.text == original.text, "Paste input changed.")
        try save(source: card.memo.source, memo: card.memo, to: directory.appending(path: "card-paste.json"))
        let previous = store.state.cards
        var invalid = ProductDraft()
        invalid.address = "https://example.invalid/product"
        store.send(.draftChanged(invalid))
        store.send(.submit(Date()))
        await harness.settle()
        guard case .failed = store.state.phase else { throw ProductError.invalidEvidence }
        try require(store.state.cards == previous && store.state.canAdd, "Failure discarded cards or blocked retry.")
    }

    private static func failure(in state: ProductState) -> String {
        if case .failed(let message) = state.phase { return message }
        return "The Store did not produce the expected card."
    }

    private static func save(source: ProductSource, memo: ProductMemo?, to url: URL) throws {
        let values = memo?.evidence.map {
            Candidate(
                field: $0.candidate.field, text: $0.candidate.text, lower: $0.candidate.range.lowerBound,
                upper: $0.candidate.range.upperBound, probability: $0.candidate.probability, context: $0.context)
        }
        let record = Record(
            title: source.title, url: source.url?.absoluteString, capturedAt: source.capturedAt,
            source: source.text, scalarCount: source.text.unicodeScalars.count,
            sha256: SHA256.hash(data: Data(source.text.utf8)).map { String(format: "%02x", $0) }.joined(),
            modelRan: memo != nil, candidates: values)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(record).write(to: url, options: .atomic)
    }

    private static func require(_ condition: Bool, _ message: String) throws {
        if !condition {
            throw NSError(domain: "ProductScenario", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
        }
    }

    private struct Record: Encodable {
        let title: String
        let url: String?
        let capturedAt: Date
        let source: String
        let scalarCount: Int
        let sha256: String
        let modelRan: Bool
        let candidates: [Candidate]?
    }

    private struct Candidate: Encodable {
        let field: ProductField
        let text: String
        let lower: Int
        let upper: Int
        let probability: Float
        let context: String
    }
}

@MainActor
private final class ProductScenarioHarness {
    let store: Store<ProductFeature>
    private let completion: ProductScenarioCompletion

    init(folder: URL) {
        let completion = ProductScenarioCompletion()
        self.completion = completion
        let runner = ProductRunner(service: LiveProductService())
        store = Store(initialState: ProductState(modelFolder: folder)) { effect, send in
            await runner.run(effect, send: send)
            completion.signal()
        }
    }

    func settle() async {
        while store.state.busy { await completion.wait() }
    }
}

@MainActor
private final class ProductScenarioCompletion {
    private var pending = 0
    private var waiting: [CheckedContinuation<Void, Never>] = []
    func signal() {
        if waiting.isEmpty { pending += 1 } else { waiting.removeFirst().resume() }
    }
    func wait() async {
        if pending > 0 { pending -= 1 } else { await withCheckedContinuation { waiting.append($0) } }
    }
}
