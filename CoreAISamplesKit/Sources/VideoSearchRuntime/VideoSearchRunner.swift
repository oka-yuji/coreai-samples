import EmbeddingCore
import Foundation
import VideoSearchFeature

public struct VideoSearchReleaseError: LocalizedError, Equatable, Sendable {
    public let detail: String
    public init(detail: String) { self.detail = detail }
    public var errorDescription: String? { "Video search could not release its model. Try stopping again." }
}

@MainActor
public final class VideoSearchRunner {
    private enum Closing {
        case running(through: Int, waiters: [CheckedContinuation<VideoSearchReleaseError?, Never>])
        case finished(through: Int, failure: VideoSearchReleaseError?)
    }

    private let service: any VideoSearchService
    private let isPermitted: @MainActor () -> Bool
    private var closing: Closing?

    public init(service: any VideoSearchService, isPermitted: @escaping @MainActor () -> Bool = { true }) {
        self.service = service
        self.isPermitted = isPermitted
    }

    public func run(_ effect: VideoSearchEffect, send: (VideoSearchAction) -> Void) async {
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
        case .load(let value, _), .index(let value, _), .search(let value, _), .export(let value, _, _, _):
            id = value
        }
        guard isPermitted() else {
            send(.failed(id, VideoSearchFailure(message: "The selected model is not available right now. Try again.")))
            return
        }
        let start = ContinuousClock.now
        do {
            switch effect {
            case .load(let id, let folder):
                try await service.load(folder)
                try Task.checkCancellation()
                send(.loaded(id))
            case .index(let id, let video):
                let index = try await service.index(video) { completed, total in
                    if !Task.isCancelled { send(.progress(id, completed, total)) }
                }
                try Task.checkCancellation()
                send(.indexed(id, index.moments, index.duration, Self.seconds(since: start)))
            case .search(let id, let query):
                let matches = try await service.search(query)
                try Task.checkCancellation()
                send(.searched(id, matches, Self.seconds(since: start)))
            case .export(let id, let plan, let video, let destination):
                let seconds = try await service.export(plan, of: video, to: destination) { value in
                    if !Task.isCancelled { send(.exportProgress(id, value)) }
                }
                try Task.checkCancellation()
                send(.exported(id, destination, seconds))
            case .waitForIdle, .close:
                return
            }
        } catch {
            guard !Task.isCancelled && !(error is CancellationError) else { return }
            if case .export = effect {
                send(.failed(id, VideoSearchFailure(message: "The clip could not be exported.", detail: "\(error)")))
            } else {
                send(.failed(id, Self.failure(error)))
            }
        }
    }

    public func release(_ request: Int) async throws {
        if let failure = await released(request) { throw failure }
    }

    private func released(_ request: Int) async -> VideoSearchReleaseError? {
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
        var failure: VideoSearchReleaseError?
        do {
            try await service.close()
        } catch {
            failure = VideoSearchReleaseError(detail: String(describing: error))
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

    private static func failure(_ error: any Error) -> VideoSearchFailure {
        let message: String
        switch error {
        case EmbeddingError.tooManyTokens: message = "Shorten your search. This model accepts up to 128 tokens."
        case EmbeddingError.invalidBundle:
            message = "This model folder is incomplete or incompatible. Open a valid EmbeddingGemma 2 bundle."
        case VideoSearchError.invalidVideo: message = "This video could not be read. Choose another video."
        default: message = "The operation failed. Try again or choose the video again."
        }
        return VideoSearchFailure(message: message, detail: String(describing: error))
    }
}
