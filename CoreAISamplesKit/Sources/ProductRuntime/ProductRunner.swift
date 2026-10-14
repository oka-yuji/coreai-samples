import Foundation
import ProductCore
import ProductFeature

@MainActor
public protocol ProductService {
    func source(draft: ProductDraft, capturedAt: Date) async throws -> ProductSource
    func extract(source: ProductSource, folder: URL) async throws -> ProductMemo
    func waitForIdle() async
}

@MainActor
public final class LiveProductService: ProductService {
    private let page = ProductPageClient()
    private let model = ProductExtractor()
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    public init() {}

    public func source(draft: ProductDraft, capturedAt: Date) async throws -> ProductSource {
        try begin()
        defer { finish() }
        if draft.mode == .text {
            return try ProductSource(title: draft.title, text: draft.text, url: nil, capturedAt: capturedAt)
        }
        return try await page.read(url: ProductPage.validatedURL(draft.address), capturedAt: capturedAt)
    }

    public func extract(source: ProductSource, folder: URL) async throws -> ProductMemo {
        try begin()
        defer { finish() }
        return try await model.extract(source: source, folder: folder)
    }

    public func waitForIdle() async {
        if busy { await withCheckedContinuation { waiting.append($0) } }
    }

    private func begin() throws {
        guard !busy else { throw ProductError.busy }
        busy = true
    }

    private func finish() {
        busy = false
        let continuations = waiting
        waiting.removeAll()
        for continuation in continuations { continuation.resume() }
    }
}

@MainActor
public struct ProductRunner {
    private let service: any ProductService
    public init(service: any ProductService) { self.service = service }

    public func run(_ effect: ProductEffect, send: (ProductAction) -> Void) async {
        let id: Int
        switch effect {
        case .load(let value, _, _), .extract(let value, _, _), .waitForIdle(let value): id = value
        }
        do {
            switch effect {
            case .load(let id, let draft, let date):
                let source = try await service.source(draft: draft, capturedAt: date)
                try Task.checkCancellation()
                send(.sourceLoaded(id, source))
            case .extract(let id, let source, let folder):
                let memo = try await service.extract(source: source, folder: folder)
                try Task.checkCancellation()
                send(.extracted(id, memo))
            case .waitForIdle(let id):
                await service.waitForIdle()
                try Task.checkCancellation()
                send(.stopped(id))
            }
        } catch {
            if !Task.isCancelled && !(error is CancellationError) { send(.failed(id, error.localizedDescription)) }
        }
    }
}
