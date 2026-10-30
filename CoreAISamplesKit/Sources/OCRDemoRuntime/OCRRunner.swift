import Foundation
import OCRFeature
import OCRRuntime

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
            send(error is CancellationError ? .cancelled(session) : .failed(session, Self.message(error, for: effect)))
        }
    }

    public func release() async { await service.release() }

    static func message(_ error: any Error, for effect: OCREffect) -> String {
        if (error as? OCRRuntimeError) == .busy { return "Another read is still finishing. Try again shortly." }
        if let message = (error as? any LocalizedError)?.errorDescription { return message }
        if error is CocoaError { return error.localizedDescription }
        switch effect {
        case .select: return "The image could not be opened. Choose another image."
        case .load: return "Model preparation failed. Check the model folder and try again."
        case .read: return "The document could not be read. Try another image."
        case .release: return "OCR could not finish stopping. Try again."
        }
    }
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
