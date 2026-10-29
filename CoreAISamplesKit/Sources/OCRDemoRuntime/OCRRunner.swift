import Foundation
import OCRFeature

@MainActor
public protocol OCRService {
    func image(at url: URL) async throws -> OCRInput
    func load(bundle: URL) async throws
    func read(_ input: OCRInput) async throws -> OCRDocument
    func release() async
}

@MainActor
public final class OCRRunner {
    private let service: any OCRService
    private let isPermitted: @MainActor () -> Bool

    public init(service: any OCRService, isPermitted: @escaping @MainActor () -> Bool = { true }) {
        self.service = service
        self.isPermitted = isPermitted
    }

    public func run(_ effect: OCREffect, send: (OCRAction) -> Void) async {
        if case .release(let session) = effect {
            await service.release()
            if !Task.isCancelled { send(.stopped(session)) }
            return
        }
        let session: Int
        switch effect {
        case .select(let id, _), .load(let id, _), .read(let id, _), .release(let id): session = id
        }
        guard isPermitted() else {
            send(.failed(session, "The selected model is not available right now. Try again."))
            return
        }
        do {
            let action: OCRAction
            switch effect {
            case .select(_, let url): action = .selected(session, try await service.image(at: url))
            case .load(_, let bundle):
                try await service.load(bundle: bundle)
                action = .loaded(session)
            case .read(_, let input): action = .completed(session, try await service.read(input))
            case .release: return
            }
            try Task.checkCancellation()
            guard isPermitted() else {
                await service.release()
                send(.cancelled(session))
                return
            }
            send(action)
        } catch {
            guard !Task.isCancelled else { return }
            send(error is CancellationError ? .cancelled(session) : .failed(session, error.localizedDescription))
        }
    }

    public func release() async { await service.release() }
}

@MainActor
public final class LiveOCRService: OCRService {
    private let pipeline: OCRPipeline

    public init() { pipeline = OCRPipeline() }
    init(pipeline: OCRPipeline) { self.pipeline = pipeline }

    public func image(at url: URL) async throws -> OCRInput { try await pipeline.image(at: url) }
    public func load(bundle: URL) async throws { try await pipeline.load(bundle: bundle) }
    public func read(_ input: OCRInput) async throws -> OCRDocument { try await pipeline.read(input) }
    public func release() async { await pipeline.release() }
}
