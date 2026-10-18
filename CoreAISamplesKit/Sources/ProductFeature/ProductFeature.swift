import FeatureStore
import Foundation
import ProductCore

public struct ProductDraft: Equatable, Sendable {
    public enum Mode: String, CaseIterable, Sendable { case url, text }
    public var mode: Mode
    public var address = ""
    public var title = ""
    public var text = ""

    public init(mode: Mode = .url) { self.mode = mode }
    public var input: String { mode == .url ? address : text }
}

public struct ProductCard: Identifiable, Equatable, Sendable {
    public let id: Int
    public let memo: ProductMemo
    public let resourceIdentity: String?
    public init(id: Int, memo: ProductMemo, resourceIdentity: String? = nil) {
        self.id = id
        self.memo = memo
        self.resourceIdentity = resourceIdentity
    }
}

public struct ProductState: Equatable, Sendable {
    public enum Phase: Equatable, Sendable {
        case idle, importing, extracting, stopping
        case failed(String)
    }
    public static let cardLimit = 4
    public var draft = ProductDraft()
    public var cards: [ProductCard] = []
    public var modelFolder: URL?
    public let resourceIdentity: String?
    public var importedSource: ProductSource?
    public var phase = Phase.idle
    public var generation = 0
    public var active = true
    public var preview = false

    public init(modelFolder: URL? = nil) {
        self.modelFolder = modelFolder
        resourceIdentity = nil
    }

    public init(folder: URL, resourceIdentity: String, keeping previous: ProductState? = nil) {
        modelFolder = folder
        self.resourceIdentity = resourceIdentity
        if let previous {
            draft = previous.draft
            cards = previous.cards
            generation = previous.generation
        }
    }

    public var managed: Bool { resourceIdentity != nil }
    public var busy: Bool {
        switch phase {
        case .importing, .extracting, .stopping: true
        case .idle, .failed: false
        }
    }
    public var canAdd: Bool {
        active && !preview && !busy && modelFolder != nil && cards.count < Self.cardLimit
            && !draft.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

public enum ProductAction: Sendable {
    case draftChanged(ProductDraft)
    case modelChosen(URL)
    case submit(Date)
    case sourceLoaded(Int, ProductSource)
    case extracted(Int, ProductMemo)
    case failed(Int, String)
    case remove(Int)
    case stop, appeared, closed
    case stopped(Int)
    case released(Int)
    case releaseFailed(Int, String)
}

public enum ProductEffect: Equatable, Sendable {
    case load(Int, ProductDraft, Date)
    case extract(Int, ProductSource, URL)
    case waitForIdle(Int)
    case release(Int)
}

public enum ProductFeature: Feature {
    public enum EffectID: Sendable { case operation, drain }

    public static func reduce(_ state: inout ProductState, _ action: ProductAction) -> [Command<
        ProductEffect, EffectID
    >] {
        switch action {
        case .draftChanged(let draft):
            guard state.draft != draft else { return [] }
            state.draft = draft
            state.importedSource = nil
            return stop(&state)
        case .modelChosen(let url):
            guard !state.managed else { return [] }
            state.modelFolder = url
            return stop(&state)
        case .submit(let date):
            guard state.canAdd else { return [] }
            state.generation += 1
            state.phase = .importing
            state.importedSource = nil
            return [.run(.load(state.generation, state.draft, date), id: .operation)]
        case .sourceLoaded(let id, let source):
            guard state.active, id == state.generation, state.phase == .importing, let folder = state.modelFolder else {
                return []
            }
            state.importedSource = source
            state.phase = .extracting
            return [.run(.extract(id, source, folder), id: .operation)]
        case .extracted(let id, let memo):
            guard state.active, id == state.generation, state.phase == .extracting,
                state.cards.count < ProductState.cardLimit
            else { return [] }
            state.cards.append(ProductCard(id: id, memo: memo, resourceIdentity: state.resourceIdentity))
            state.phase = .idle
            state.importedSource = nil
            state.draft = ProductDraft(mode: state.draft.mode)
            return []
        case .failed(let id, let message):
            guard state.active, id == state.generation, state.phase == .importing || state.phase == .extracting else {
                return []
            }
            state.phase = .failed(message)
            return []
        case .remove(let id):
            state.cards.removeAll { $0.id == id }
            return []
        case .stop:
            return stop(&state)
        case .stopped(let id):
            guard id == state.generation, state.phase == .stopping else { return [] }
            state.phase = .idle
            return []
        case .appeared:
            state.active = true
            return []
        case .closed:
            state.active = false
            state.generation += 1
            state.importedSource = nil
            state.phase = .stopping
            return [.cancelAll, .run(.release(state.generation), id: .drain)]
        case .released(let id):
            finishRelease(&state, id, as: .idle)
            return []
        case .releaseFailed(let id, let message):
            finishRelease(&state, id, as: .failed(message))
            return []
        }
    }

    private static func finishRelease(_ state: inout ProductState, _ id: Int, as phase: ProductState.Phase) {
        if id == state.generation, state.phase == .stopping { state.phase = phase }
    }

    private static func stop(_ state: inout ProductState) -> [Command<ProductEffect, EffectID>] {
        guard state.phase == .importing || state.phase == .extracting else { return [] }
        state.generation += 1
        state.phase = .stopping
        state.importedSource = nil
        return [.cancel(.operation), .run(.waitForIdle(state.generation), id: .drain)]
    }
}
