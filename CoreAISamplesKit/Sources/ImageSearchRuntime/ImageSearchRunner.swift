import EmbeddingCore
import Foundation
import ImageSearchFeature

public struct ImageSearchReleaseError: LocalizedError, Equatable, Sendable {
    public let detail: String
    public init(detail: String) { self.detail = detail }
    public var errorDescription: String? { "Image search could not release its model. Try stopping again." }
}

@MainActor
public final class ImageSearchRunner {
    private enum Closing {
        case running(through: Int, waiters: [CheckedContinuation<ImageSearchReleaseError?, Never>])
        case finished(through: Int, failure: ImageSearchReleaseError?)
    }

    private let service: any ImageSearchService
    private let isPermitted: @MainActor () -> Bool
    private var closing: Closing?

    public init(service: any ImageSearchService, isPermitted: @escaping @MainActor () -> Bool = { true }) {
        self.service = service
        self.isPermitted = isPermitted
    }

    public func run(_ effect: ImageSearchEffect, send: (ImageSearchAction) -> Void) async {
        let id: Int
        switch effect {
        case .waitForIdle(let value):
            await service.waitForIdle()
            if !Task.isCancelled { send(.stopped(value)) }
            return
        case .close(let value):
            let failure = await released(value)
            if !Task.isCancelled { send(failure.map { .closeFailed(value, $0.detail) } ?? .stopped(value)) }
            return
        case .load(let value, _), .index(let value, _), .search(let value, _):
            id = value
        }
        guard isPermitted() else {
            send(.failed(id, ImageSearchFailure(message: "The selected model is not available right now. Try again.")))
            return
        }
        let start = ContinuousClock.now
        do {
            switch effect {
            case .load(let id, let folder):
                try await service.load(folder)
                try Task.checkCancellation()
                send(.loaded(id))
            case .index(let id, let urls):
                let photos = try await service.index(urls) { count in
                    if !Task.isCancelled { send(.progress(id, count)) }
                }
                try Task.checkCancellation()
                send(.indexed(id, photos, Self.seconds(since: start)))
            case .search(let id, let query):
                let matches = try await service.search(query)
                try Task.checkCancellation()
                send(.searched(id, matches, Self.seconds(since: start)))
            case .waitForIdle, .close:
                return
            }
        } catch {
            if !Task.isCancelled && !(error is CancellationError) { send(.failed(id, Self.failure(error))) }
        }
    }

    public func release(_ request: Int) async throws {
        if let failure = await released(request) { throw failure }
    }

    private func released(_ request: Int) async -> ImageSearchReleaseError? {
        switch closing {
        case .running(let through, let waiters)?:
            return await withCheckedContinuation {
                closing = .running(through: max(through, request), waiters: waiters + [$0])
            }
        case .finished(let through, let failure)? where request <= through:
            return failure
        case .finished?, nil:
            break
        }
        closing = .running(through: request, waiters: [])
        var failure: ImageSearchReleaseError?
        do {
            try await service.close()
        } catch {
            failure = ImageSearchReleaseError(detail: String(describing: error))
        }
        guard case .running(let through, let waiters)? = closing else { return failure }
        closing = .finished(through: through, failure: failure)
        for waiter in waiters { waiter.resume(returning: failure) }
        return failure
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = (ContinuousClock.now - start).components
        return Double(duration.seconds) + Double(duration.attoseconds) / 1e18
    }

    private static func failure(_ error: any Error) -> ImageSearchFailure {
        let message: String
        switch error {
        case EmbeddingError.tooManyTokens: message = "Shorten your search. This model accepts up to 128 tokens."
        case EmbeddingError.invalidImage: message = "Use an opaque, still image with no more than 40 million pixels."
        case EmbeddingError.invalidBundle:
            message = "This model folder is incomplete or incompatible. Open a valid EmbeddingGemma 2 bundle."
        case let error as CocoaError where error.code == .fileReadNoSuchFile:
            message = "Some images can no longer be opened. Choose the images again."
        default: message = "The operation failed. Try again or reopen the model folder."
        }
        return ImageSearchFailure(message: message, detail: String(describing: error))
    }
}
