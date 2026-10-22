import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import os

@MainActor
public final class ModelLibraryResourceRunner {
    private struct Activity {
        let ticket: LibraryTicket
        var joiners: [CheckedContinuation<Void, Never>] = []
    }

    private let store: ManagedModelStore
    private let transfer: ArtifactTransfer
    private let migrationCandidates: @Sendable () -> [URL]
    private let logger = Logger(subsystem: "com.coreai-samples", category: "ModelLibrary")
    private var activities: [String: Activity] = [:]

    public init(
        store: ManagedModelStore, transfer: ArtifactTransfer,
        migrationCandidates: @escaping @Sendable () -> [URL] = { [] }
    ) {
        self.store = store
        self.transfer = transfer
        self.migrationCandidates = migrationCandidates
    }

    public func run(_ effect: ModelLibraryEffect, send: (ModelLibraryAction) -> Void) async {
        switch effect {
        case .inspect(let model, let artifact, let ticket):
            await exclusively(artifact.id, ticket, send) { [self] send in
                let resource = try await inspect(model, artifact)
                send(.inspected(artifact.id, ticket, resource))
            }
        case .download(let model, let artifact, let ticket):
            await exclusively(artifact.id, ticket, send) { [self] send in
                let staging = try await store.stagingDirectory(for: model, artifact: artifact)
                let (updates, sink) = AsyncStream.makeStream(of: ModelTransferProgress.self)
                async let transferred: Void = Self.relay(transfer, model, artifact, staging, sink)
                for await update in updates { send(.progress(artifact.id, ticket, update)) }
                try await transferred
                send(.downloaded(artifact.id, ticket, staging))
            }
        case .importBundle(let model, let artifact, let ticket, let source):
            await exclusively(artifact.id, ticket, send, failure: validationFailure) { [self] send in
                let (updates, sink) = AsyncStream.makeStream(
                    of: ModelTransferProgress.self, bufferingPolicy: .bufferingNewest(1))
                async let copied = Self.relayImport(store, model, artifact, source, sink)
                for await update in updates { send(.progress(artifact.id, ticket, update)) }
                let staging = try await copied
                do {
                    try Task.checkCancellation()
                    send(.importCopied(artifact.id, ticket))
                    try Task.checkCancellation()
                    let resource = try await store.promote(staging: staging, model: model, artifact: artifact)
                    send(.validated(artifact.id, ticket, resource))
                } catch {
                    try? await store.discardImport(staging: staging, model: model, artifact: artifact)
                    throw error
                }
            }
        case .validate(let model, let artifact, let ticket, let staging):
            await exclusively(artifact.id, ticket, send, failure: validationFailure) { [self] send in
                do {
                    let resource = try await store.promote(staging: staging, model: model, artifact: artifact)
                    send(.validated(artifact.id, ticket, resource))
                } catch {
                    if artifact.sourceKind == .localManifest {
                        try? await store.discardImport(staging: staging, model: model, artifact: artifact)
                    }
                    throw error
                }
            }
        case .delete(let resource, let ticket):
            await exclusively(resource.artifactID, ticket, send, failure: deletionFailure) { [self] send in
                _ = try await store.delete(identity: resource.identity)
                send(.deleted(resource.artifactID, ticket))
            }
        case .stopTransfer(let artifactID, let ticket):
            if activities[artifactID]?.ticket == ticket {
                await withCheckedContinuation { activities[artifactID]?.joiners.append($0) }
            }
            send(.transferStopped(artifactID, ticket))
        case .stopImport(let model, let artifact, let ticket):
            if activities[artifact.id]?.ticket == ticket {
                await withCheckedContinuation { activities[artifact.id]?.joiners.append($0) }
            }
            let resource: ValidatedModelResource?
            do {
                resource = try await inspect(model, artifact)
            } catch {
                resource = nil
                logger.error(
                    "Import stop inspection rejected \(artifact.id, privacy: .public): \(String(describing: error))")
            }
            if !Task.isCancelled { send(.importStopped(artifact.id, ticket, resource)) }
        case .stopDemo(let session, let ticket):
            send(.demoStopFailed(session.id, ticket, "Demo execution is not connected"))
        case .prepareDemo(let session):
            send(.demoPreparationChanged(session.id, .unavailable))
        }
    }

    private func exclusively(
        _ artifactID: String, _ ticket: LibraryTicket, _ send: (ModelLibraryAction) -> Void,
        failure: (any Error) -> ModelLibraryFailure = { .operation(describe($0)) },
        _ operation: @escaping ((ModelLibraryAction) -> Void) async throws -> Void
    ) async {
        func report(_ action: ModelLibraryAction) {
            if !Task.isCancelled { send(action) }
        }
        guard activities[artifactID] == nil else {
            report(.failed(artifactID, ticket, .operation("transfer in progress")))
            return
        }
        activities[artifactID] = Activity(ticket: ticket)
        defer {
            for joiner in activities.removeValue(forKey: artifactID)?.joiners ?? [] { joiner.resume() }
        }
        do {
            try await operation(report)
        } catch {
            guard !(error is CancellationError) else { return }
            report(.failed(artifactID, ticket, failure(error)))
        }
    }

    private func inspect(_ model: LibraryModel, _ artifact: ModelArtifact) async throws -> ValidatedModelResource? {
        switch try await store.inspect(model: model, artifact: artifact) {
        case .ready(let resource):
            return resource
        case .invalid(let error):
            logger.error("Inspection rejected \(artifact.id, privacy: .public): \(String(describing: error))")
            return nil
        case .missing:
            guard artifact.sourceKind == .huggingFace else { return nil }
            let candidates = migrationCandidates()
            guard !candidates.isEmpty else { return nil }
            let outcomes = await store.migrate(candidates: candidates, model: model, artifact: artifact).outcomes
            for case .migrated(let resource)? in candidates.map({ outcomes[$0] }) { return resource }
            for case .failed(let error)? in candidates.map({ outcomes[$0] }) { throw error }
            return nil
        }
    }

    nonisolated private static func relay(
        _ transfer: ArtifactTransfer, _ model: LibraryModel, _ artifact: ModelArtifact, _ staging: URL,
        _ sink: AsyncStream<ModelTransferProgress>.Continuation
    ) async throws {
        defer { sink.finish() }
        try await transfer.download(model: model, artifact: artifact, to: staging) { sink.yield($0) }
    }

    nonisolated private static func relayImport(
        _ store: ManagedModelStore, _ model: LibraryModel, _ artifact: ModelArtifact, _ source: URL,
        _ sink: AsyncStream<ModelTransferProgress>.Continuation
    ) async throws -> URL {
        defer { sink.finish() }
        return try await store.copyForImport(from: source, model: model, artifact: artifact) { sink.yield($0) }
    }
}

private func validationFailure(_ error: any Error) -> ModelLibraryFailure {
    switch error as? ManagedModelStoreError {
    case .inUse: .modelInUse
    case .insufficientSpace, .ledgerCorrupted: .operation(describe(error))
    case .some: .invalidResource
    case nil: error is CocoaError ? .operation(describe(error)) : .invalidResource
    }
}

private func deletionFailure(_ error: any Error) -> ModelLibraryFailure {
    error as? ManagedModelStoreError == .inUse ? .modelInUse : .operation(describe(error))
}

private func describe(_ error: any Error) -> String {
    error is ManagedModelStoreError ? String(describing: error) : error.localizedDescription
}
