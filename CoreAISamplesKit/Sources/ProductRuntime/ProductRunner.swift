import Foundation
import ProductCore
import ProductFeature

@MainActor
public protocol ProductService {
    func source(draft: ProductDraft, capturedAt: Date) async throws -> ProductSource
    func extract(source: ProductSource, folder: URL) async throws -> ProductMemo
    func waitForIdle() async
    func release() async throws
}

@MainActor
public final class LiveProductService: ProductService {
    private let page: ProductPageClient
    private let model = ProductExtractor()
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    public init() { page = ProductPageClient() }

    init(page: ProductPageClient) { self.page = page }

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

    public func release() async throws {
        guard !busy else { throw ProductError.busy }
        try await model.release()
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

struct ProductExecutionClosed: LocalizedError {
    var errorDescription: String? { "The selected model is not ready yet. Please try again shortly." }
}

@MainActor
public struct ProductRunner {
    private let service: any ProductService
    private let permits: @MainActor () -> Bool

    public init(service: any ProductService, permits: @escaping @MainActor () -> Bool = { true }) {
        self.service = service
        self.permits = permits
    }

    public func run(_ effect: ProductEffect, send: (ProductAction) -> Void) async {
        do {
            switch effect {
            case .load(let id, let draft, let date):
                guard permits() else { throw ProductExecutionClosed() }
                let source = try await service.source(draft: draft, capturedAt: date)
                try Task.checkCancellation()
                send(.sourceLoaded(id, source))
            case .extract(let id, let source, let folder):
                guard permits() else { throw ProductExecutionClosed() }
                let memo = try await service.extract(source: source, folder: folder)
                try Task.checkCancellation()
                send(.extracted(id, memo))
            case .waitForIdle(let id):
                await service.waitForIdle()
                try Task.checkCancellation()
                send(.stopped(id))
            case .release(let id):
                try await release()
                try Task.checkCancellation()
                send(.released(id))
            }
        } catch {
            guard !Task.isCancelled, !(error is CancellationError) else { return }
            switch effect {
            case .load(let id, _, _), .extract(let id, _, _), .waitForIdle(let id):
                send(.failed(id, error.localizedDescription))
            case .release(let id): send(.releaseFailed(id, error.localizedDescription))
            }
        }
    }

    public func release() async throws {
        await service.waitForIdle()
        try await service.release()
    }
}
