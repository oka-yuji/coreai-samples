import EmbeddingCore
import Foundation
import ImageSearchFeature

@MainActor
public struct ImageSearchRunner {
    private let service: any ImageSearchService
    public init(service: any ImageSearchService) { self.service = service }

    public func run(_ effect: ImageSearchEffect, send: (ImageSearchAction) -> Void) async {
        let id: Int
        switch effect {
        case .load(let value, _), .index(let value, _), .search(let value, _), .waitForIdle(let value),
            .close(let value):
            id = value
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
            case .waitForIdle(let id):
                await service.waitForIdle()
                try Task.checkCancellation()
                send(.stopped(id))
            case .close(let id):
                try await service.close()
                try Task.checkCancellation()
                send(.stopped(id))
            }
        } catch {
            if !Task.isCancelled && !(error is CancellationError) { send(.failed(id, Self.failure(error))) }
        }
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
        default: message = "The operation failed. Try again or reopen the model folder."
        }
        return ImageSearchFailure(message: message, detail: String(describing: error))
    }
}
