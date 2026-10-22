import FeatureStore
import Foundation
import ImageGenerationCore

public enum ImageGenerationAction: Sendable {
    case appeared, ended
    case setPrompt(String)
    case setSeed(String)
    case chooseReference(URL)
    case referenceSelectionFailed
    case referencePrepared(ImageGenerationTicket, ImageReferencePreview)
    case confirmReference(ImageGenerationTicket)
    case dismissReference(ImageGenerationTicket)
    case removeReference
    case generate(id: UUID, randomSeed: UInt64)
    case progressed(ImageGenerationTicket, ImageExecutionProgress)
    case generated(ImageGenerationTicket, ImageGeneratedAsset)
    case failed(ImageGenerationTicket, ImageGenerationFailure)
    case cancelled(ImageGenerationTicket)
    case stop
    case stopped(ImageGenerationTicket)
    case stopFailed(ImageGenerationTicket)
    case prepareExport(UUID)
    case exportDismissed(ImageExportTicket)
    case exportCancelled(ImageExportTicket)
    case exportFinished(ImageExportTicket, Bool)
    case dismissFailure
}

public enum ImageGenerationEffect: Equatable, Sendable {
    case reference(ImageGenerationTicket, URL)
    case generate(ImageGenerationTicket, ImageGenerationRequest)
    case stop(ImageGenerationTicket)
}

public enum ImageGenerationEffectID: Hashable, Sendable { case operation, stop }

public enum ImageGenerationFeature: Feature {
    public typealias State = ImageGenerationState
    public typealias Action = ImageGenerationAction
    public typealias Effect = ImageGenerationEffect
    public typealias EffectID = ImageGenerationEffectID
    public typealias Command = FeatureStore.Command<Effect, EffectID>

    public static func reduce(_ state: inout State, _ action: Action) -> [Command] {
        if !state.presented {
            switch action {
            case .appeared, .stopped, .stopFailed, .exportDismissed, .exportCancelled, .exportFinished: break
            default: return []
            }
        }
        switch action {
        case .appeared:
            state.presented = true
        case .ended:
            state.presented = false
            state.proposedReference = nil
            state.phase = .stopping
            state.progress = nil
            state.requestID = nil
            let ticket = advance(&state)
            return [.cancelAll, .run(.stop(ticket), id: .stop)]
        case .setPrompt(let prompt):
            guard state.editable else { return [] }
            state.prompt = prompt
            invalidate(&state)
        case .setSeed(let seed):
            guard state.editable else { return [] }
            state.seed = seed
            invalidate(&state)
        case .chooseReference(let url):
            guard state.editable else { return [] }
            let ticket = advance(&state)
            state.phase = .readingReference
            state.failure = nil
            return [.run(.reference(ticket, url), id: .operation)]
        case .referenceSelectionFailed:
            guard state.editable else { return [] }
            state.failure = .invalidImage
        case .referencePrepared(let ticket, let preview):
            guard ticket == state.operation, preview.id == ticket, state.phase == .readingReference else { return [] }
            guard preview.pixels.width == 512, preview.pixels.height == 512 else {
                state.phase = .idle
                state.failure = .invalidImage
                return []
            }
            state.phase = .confirmingReference
            state.proposedReference = preview
        case .confirmReference(let ticket):
            guard state.phase == .confirmingReference, let preview = state.proposedReference, preview.id == ticket
            else {
                return []
            }
            state.reference = preview
            state.proposedReference = nil
            state.phase = .idle
            invalidate(&state)
        case .dismissReference(let ticket):
            guard state.proposedReference?.id == ticket, state.phase == .confirmingReference else { return [] }
            state.proposedReference = nil
            state.phase = .idle
        case .removeReference:
            guard state.editable else { return [] }
            state.reference = nil
            invalidate(&state)
        case .generate(let id, let randomSeed):
            guard state.canGenerate else { return [] }
            let seed = UInt64(state.seed.trimmingCharacters(in: .whitespacesAndNewlines)) ?? randomSeed
            guard seed <= Int64.max else { return [] }
            let ticket = advance(&state)
            state.seed = String(seed)
            state.phase = .generating
            state.failure = nil
            state.progress = .preparing
            state.requestID = id
            state.export = nil
            state.exportSucceeded = false
            return [
                .run(
                    .generate(
                        ticket,
                        ImageGenerationRequest(
                            id: id, prompt: state.prompt, seed: seed, reference: state.reference?.pixels)),
                    id: .operation)
            ]
        case .progressed(let ticket, let progress):
            guard ticket == state.operation, state.phase == .generating, progress.requestID == state.requestID else {
                return []
            }
            state.progress = progress.phase
        case .generated(let ticket, let result):
            guard ticket == state.operation, state.phase == .generating,
                result.id == state.requestID, result.report.requestID == result.id, result.report.status == .completed,
                result.seed == UInt64(state.seed)
            else { return [] }
            state.result = result
            state.phase = .idle
            state.progress = nil
        case .failed(let ticket, let failure):
            guard ticket == state.operation, [.generating, .readingReference].contains(state.phase) else { return [] }
            state.phase = failure == .cleanupFailed ? .cleanupFailed : .idle
            state.failure = failure
            state.progress = nil
        case .cancelled(let ticket):
            guard ticket == state.operation, [.generating, .readingReference].contains(state.phase) else { return [] }
            state.phase = .idle
            state.failure = nil
            state.progress = nil
            state.requestID = nil
        case .stop:
            guard state.canStop else { return [] }
            state.phase = .stopping
            state.progress = nil
            state.requestID = nil
            let ticket = advance(&state)
            return [.cancel(.operation), .run(.stop(ticket), id: .stop)]
        case .stopped(let ticket):
            guard ticket == state.operation, state.phase == .stopping else { return [] }
            state.phase = .idle
            state.failure = nil
        case .stopFailed(let ticket):
            guard ticket == state.operation, state.phase == .stopping else { return [] }
            state.phase = .cleanupFailed
            state.failure = .cleanupFailed
        case .prepareExport(let id):
            guard state.canExport, let result = state.result else { return [] }
            state.export = ImageExportTicket(id: id, resultID: result.id)
            state.exportDismissed = false
            state.exportSucceeded = false
            if state.failure == .exportFailed { state.failure = nil }
        case .exportDismissed(let ticket):
            guard state.export == ticket else { return [] }
            state.exportDismissed = true
        case .exportCancelled(let ticket):
            guard state.export == ticket else { return [] }
            state.export = nil
            state.exportSucceeded = false
        case .exportFinished(let ticket, let success):
            guard state.export == ticket, state.result?.id == ticket.resultID else { return [] }
            state.export = nil
            state.exportSucceeded = success
            if !success { state.failure = .exportFailed }
        case .dismissFailure:
            state.failure = nil
        }
        return []
    }

    private static func advance(_ state: inout State) -> ImageGenerationTicket {
        state.sequence += 1
        let ticket = ImageGenerationTicket(owner: state.owner, sequence: state.sequence)
        state.operation = ticket
        return ticket
    }

    private static func invalidate(_ state: inout State) {
        _ = advance(&state)
        state.result = nil
        state.progress = nil
        state.failure = nil
        state.requestID = nil
        state.export = nil
        state.exportSucceeded = false
    }
}
