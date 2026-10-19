import FeatureStore
import Foundation
import ProductCore
import ProductFeature
import ProductRuntime
import Testing

struct ProductFeatureTests {
    private let stamp = Date(timeIntervalSince1970: 0)
    private let selected = URL(filePath: "/models/selected")
    private let identity = "gliner2.5-multi/gliner2.5-multi-fp32@0123456789abcdef0123456789abcdef01234567"

    private func ready() -> ProductState {
        var state = ProductState(modelFolder: URL(filePath: "/models/product"))
        state.draft.mode = .text
        state.draft.text = "Memory: 16 GB"
        return state
    }

    private func managed() -> ProductState {
        var state = ProductState(folder: selected, resourceIdentity: identity)
        state.draft.mode = .text
        state.draft.text = "Memory: 16 GB"
        return state
    }

    private func source() throws -> ProductSource {
        try ProductSource(title: "Example", text: "Memory: 16 GB", url: nil, capturedAt: Date(timeIntervalSince1970: 0))
    }

    private func addCard(_ state: inout ProductState, text: String) throws {
        var draft = state.draft
        draft.text = text
        _ = ProductFeature.reduce(&state, .draftChanged(draft))
        _ = ProductFeature.reduce(&state, .submit(stamp))
        let id = state.generation
        let source = try ProductSource(title: "", text: text, url: nil, capturedAt: stamp)
        _ = ProductFeature.reduce(&state, .sourceLoaded(id, source))
        _ = ProductFeature.reduce(&state, .extracted(id, try ProductMemo(source: source, candidates: [])))
    }

    @Test("A submitted description is a stable snapshot and a duplicate submission is ignored")
    func submission() {
        var state = ready()
        let stamp = Date(timeIntervalSince1970: 0)
        #expect(ProductFeature.reduce(&state, .submit(stamp)) == [.run(.load(1, state.draft, stamp), id: .operation)])
        #expect(ProductFeature.reduce(&state, .submit(stamp)).isEmpty)
        #expect(!state.canAdd)
    }

    @Test("Source and extraction completions produce one source-backed card")
    func completedCard() throws {
        var state = ready()
        _ = ProductFeature.reduce(&state, .submit(Date(timeIntervalSince1970: 0)))
        let source = try source()
        let folder = try #require(state.modelFolder)
        #expect(
            ProductFeature.reduce(&state, .sourceLoaded(1, source)) == [
                .run(.extract(1, source, folder), id: .operation)
            ])
        let memo = try ProductMemo(source: source, candidates: [])
        _ = ProductFeature.reduce(&state, .extracted(1, memo))
        #expect(state.cards == [ProductCard(id: 1, memo: memo)])
        #expect(state.phase == .idle)
        #expect(state.draft.text.isEmpty)
        #expect(ProductFeature.reduce(&state, .extracted(1, memo)).isEmpty)
        #expect(state.cards.count == 1)
    }

    @Test("Editing the draft cancels the old request and waits for the resource to become idle")
    func editedDraft() throws {
        var state = ready()
        _ = ProductFeature.reduce(&state, .submit(Date(timeIntervalSince1970: 0)))
        var draft = state.draft
        draft.text = "Storage: 512 GB"
        #expect(
            ProductFeature.reduce(&state, .draftChanged(draft)) == [
                .cancel(.operation), .run(.waitForIdle(2), id: .drain),
            ])
        #expect(state.phase == .stopping)
        #expect(ProductFeature.reduce(&state, .sourceLoaded(1, try source())).isEmpty)
        #expect(ProductFeature.reduce(&state, .failed(1, "old failure")).isEmpty)
        _ = ProductFeature.reduce(&state, .stopped(2))
        #expect(state.draft.text == "Storage: 512 GB")
        #expect(state.canAdd)
    }

    @Test("Closing the feature cancels all work and rejects late output after reopening")
    func closes() throws {
        var state = ready()
        _ = ProductFeature.reduce(&state, .submit(Date(timeIntervalSince1970: 0)))
        _ = ProductFeature.reduce(&state, .sourceLoaded(1, try source()))
        let commands = ProductFeature.reduce(&state, .closed)
        #expect(commands == [.cancelAll, .run(.release(2), id: .drain)])
        _ = ProductFeature.reduce(&state, .appeared)
        #expect(!state.canAdd)
        let memo = try ProductMemo(source: source(), candidates: [])
        _ = ProductFeature.reduce(&state, .extracted(1, memo))
        #expect(state.cards.isEmpty)
        _ = ProductFeature.reduce(&state, .released(2))
        #expect(state.canAdd)
    }

    @Test("Failures preserve existing cards and allow a fresh retry")
    func failurePreservesCards() throws {
        var state = ready()
        let memo = try ProductMemo(source: source(), candidates: [])
        state.cards = [ProductCard(id: 0, memo: memo)]
        _ = ProductFeature.reduce(&state, .submit(Date(timeIntervalSince1970: 0)))
        _ = ProductFeature.reduce(&state, .failed(1, "Cannot read this page"))
        #expect(state.cards.count == 1)
        #expect(state.canAdd)
        #expect(!ProductFeature.reduce(&state, .submit(Date(timeIntervalSince1970: 1))).isEmpty)
        #expect(state.generation == 2)
    }

    @Test("The four-card limit and missing model prevent new work without deleting existing cards")
    func cardLimit() throws {
        var state = ready()
        let memo = try ProductMemo(source: source(), candidates: [])
        state.cards = (1...4).map { ProductCard(id: $0, memo: memo) }
        #expect(!state.canAdd)
        #expect(ProductFeature.reduce(&state, .submit(Date(timeIntervalSince1970: 0))).isEmpty)
        _ = ProductFeature.reduce(&state, .remove(2))
        #expect(state.cards.map(\.id) == [1, 3, 4])
        #expect(state.canAdd)
        state.modelFolder = nil
        #expect(!state.canAdd)
    }

    @Test("Closing during an existing stop starts a matching drain instead of leaving the feature stuck")
    func closesWhileStopping() {
        var state = ready()
        _ = ProductFeature.reduce(&state, .submit(Date(timeIntervalSince1970: 0)))
        _ = ProductFeature.reduce(&state, .stop)
        #expect(ProductFeature.reduce(&state, .closed) == [.cancelAll, .run(.release(3), id: .drain)])
        _ = ProductFeature.reduce(&state, .stopped(2))
        #expect(state.phase == .stopping)
        _ = ProductFeature.reduce(&state, .released(3))
        #expect(state.phase == .idle)
    }

    @MainActor
    @Test("The runner keeps fetched source data intact when passing it to extraction")
    func runner() async throws {
        let source = try source()
        let service = ProductServiceStub(source: source)
        let runner = ProductRunner(service: service)
        var actions: [ProductAction] = []
        var draft = ProductDraft(mode: .text)
        draft.text = source.text
        await runner.run(.load(4, draft, source.capturedAt)) { actions.append($0) }
        guard case .sourceLoaded(4, let result) = try #require(actions.first) else {
            Issue.record("The source completion was missing")
            return
        }
        #expect(result == source)
        actions.removeAll()
        await runner.run(.extract(4, result, URL(filePath: "/model"))) { actions.append($0) }
        guard case .extracted(4, let memo) = try #require(actions.first) else {
            Issue.record("The extraction completion was missing")
            return
        }
        #expect(memo.source == source)
    }

    @Test(
        "Inputs, up to four comparison results and their sources are kept when the model is deleted or the screen is rebuilt"
    )
    func retention() throws {
        var state = managed()
        let texts = ["Memory: 8 GB", "Memory: 16 GB", "Memory: 24 GB", "Memory: 32 GB"]
        for text in texts { try addCard(&state, text: text) }
        var draft = state.draft
        draft.text = "Storage: 1 TB"
        _ = ProductFeature.reduce(&state, .draftChanged(draft))
        let cards = state.cards
        #expect(cards.map(\.memo.source.text) == texts)
        #expect(cards.allSatisfy { $0.resourceIdentity == identity })
        _ = ProductFeature.reduce(&state, .closed)
        _ = ProductFeature.reduce(&state, .released(state.generation))
        #expect(state.cards == cards && state.draft == draft && state.phase == .idle)
        let other = URL(filePath: "/models/other")
        let otherIdentity = "gliner2.5-multi/gliner2.5-multi-int8@89abcdef0123456789abcdef0123456789abcdef"
        var rebuilt = ProductState(folder: other, resourceIdentity: otherIdentity, keeping: state)
        #expect(rebuilt.cards == cards && rebuilt.draft == draft && rebuilt.modelFolder == other)
        #expect(!rebuilt.canAdd)
        _ = ProductFeature.reduce(&rebuilt, .remove(cards[0].id))
        try addCard(&rebuilt, text: "Memory: 64 GB")
        #expect(rebuilt.cards.map(\.resourceIdentity) == [identity, identity, identity, otherIdentity])
        #expect(Set(rebuilt.cards.map(\.id)).count == ProductState.cardLimit)
    }

    @Test("Closing releases the model in a drain that rejects new work until its own release result arrives")
    func releaseOrdering() {
        var state = ready()
        #expect(ProductFeature.reduce(&state, .closed) == [.cancelAll, .run(.release(1), id: .drain)])
        _ = ProductFeature.reduce(&state, .appeared)
        #expect(state.phase == .stopping && !state.canAdd)
        #expect(ProductFeature.reduce(&state, .submit(stamp)).isEmpty)
        _ = ProductFeature.reduce(&state, .released(0))
        _ = ProductFeature.reduce(&state, .releaseFailed(0, "Late failure"))
        #expect(state.phase == .stopping)
        _ = ProductFeature.reduce(&state, .releaseFailed(1, "The model is still in use"))
        #expect(state.phase == .failed("The model is still in use") && state.canAdd)
        #expect(ProductFeature.reduce(&state, .closed) == [.cancelAll, .run(.release(2), id: .drain)])
        _ = ProductFeature.reduce(&state, .released(1))
        #expect(state.phase == .stopping)
        _ = ProductFeature.reduce(&state, .released(2))
        #expect(state.phase == .idle)
    }
}

@MainActor
private final class ProductServiceStub: ProductService {
    let value: ProductSource
    init(source: ProductSource) { value = source }
    func source(draft: ProductDraft, capturedAt: Date) async throws -> ProductSource { value }
    func extract(source: ProductSource, folder: URL) async throws -> ProductMemo {
        try ProductMemo(source: source, candidates: [])
    }
    func waitForIdle() async {}
    func release() async throws {}
}
