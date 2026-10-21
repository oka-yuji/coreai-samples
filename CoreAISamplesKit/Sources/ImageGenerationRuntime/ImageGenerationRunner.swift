import Foundation
import ImageGenerationCore
import ImageGenerationFeature

@MainActor
public final class ImageGenerationRunner {
    private let service: any ImageGenerationService
    private let media: ImageGenerationMedia
    private let permission: @MainActor @Sendable () -> Bool

    public init(
        service: any ImageGenerationService, media: ImageGenerationMedia = ImageGenerationMedia(),
        isPermitted: @escaping @MainActor @Sendable () -> Bool
    ) {
        self.service = service
        self.media = media
        permission = isPermitted
    }

    public func run(_ effect: ImageGenerationEffect, send: (ImageGenerationAction) -> Void) async {
        let ticket: ImageGenerationTicket
        switch effect {
        case .reference(let value, _), .generate(let value, _), .stop(let value): ticket = value
        }
        do {
            switch effect {
            case .reference(_, let url):
                let preview = try await media.reference(url, ticket: ticket)
                try Task.checkCancellation()
                send(.referencePrepared(ticket, preview))
            case .generate(_, let request):
                guard permission() else { throw ImageExecutionError.permissionDenied }
                let result = try await service.generate(request, trace: nil) { progress in
                    guard !Task.isCancelled, permission(), progress.requestID == request.id else { return }
                    send(.progressed(ticket, progress))
                }
                try Task.checkCancellation()
                guard permission() else { throw ImageExecutionError.permissionDenied }
                let asset = try await media.asset(result, request: request)
                try Task.checkCancellation()
                guard permission() else { throw ImageExecutionError.permissionDenied }
                send(.generated(ticket, asset))
            case .stop:
                try await service.stop()
                await media.join()
                try Task.checkCancellation()
                send(.stopped(ticket))
            }
        } catch {
            guard !Task.isCancelled else { return }
            if case .stop = effect {
                send(.stopFailed(ticket))
            } else if error is CancellationError {
                send(.cancelled(ticket))
            } else {
                send(.failed(ticket, Self.failure(error)))
            }
        }
    }

    public func release() async throws {
        try await service.release()
        await media.join()
    }

    private static func failure(_ error: any Error) -> ImageGenerationFailure {
        switch error {
        case ImageGenerationError.promptTooLong(let count): .promptTooLong(count)
        case ImageGenerationError.unsupportedImage: .invalidImage
        case ImageGenerationError.invalidTokens, ImageGenerationError.invalidSeed, ImageGenerationError.inputTooLarge:
            .invalidInput
        case ImageExecutionError.insufficientMemory, ImageExecutionError.memoryUnavailable: .insufficientMemory
        case ImageExecutionError.invalidResource: .invalidModel
        case ImageExecutionError.releaseUnconfirmed: .cleanupFailed
        case ImageExecutionError.permissionDenied, ImageExecutionError.released, ImageExecutionError.busy,
            ImageExecutionError.budgetNotConfigured:
            .unavailable
        default: .generationFailed
        }
    }
}
