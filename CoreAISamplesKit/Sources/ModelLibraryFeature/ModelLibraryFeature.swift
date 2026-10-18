import FeatureStore
import Foundation

public enum ModelLibraryAction: Sendable {
    case refresh
    case queryChanged(String)
    case openDemo(DemoSelection)
    case closeDemo
    case suspendDemo
    case selectArtifact(modelID: String, artifactID: String)
    case capabilityChanged(DemoCapability, CapabilityState)
    case inspected(String, LibraryTicket, ValidatedModelResource?)
    case downloadRequested(String)
    case progress(String, LibraryTicket, ModelTransferProgress)
    case downloaded(String, LibraryTicket, URL)
    case validated(String, LibraryTicket, ValidatedModelResource)
    case failed(String, LibraryTicket, ModelLibraryFailure)
    case cancelRequested(String)
    case transferStopped(String, LibraryTicket)
    case deleteRequested(String)
    case deleted(String, LibraryTicket)
    case demoStopped(LibraryTicket, LibraryTicket)
    case demoStopFailed(LibraryTicket, LibraryTicket, String)
    case retryDemoStop
    case dismissFailure
}

public enum ModelLibraryEffect: Equatable, Sendable {
    case inspect(LibraryModel, ModelArtifact, LibraryTicket)
    case download(LibraryModel, ModelArtifact, LibraryTicket)
    case validate(LibraryModel, ModelArtifact, LibraryTicket, URL)
    case stopTransfer(artifactID: String, LibraryTicket)
    case delete(ValidatedModelResource, LibraryTicket)
    case stopDemo(ModelDemoSession, LibraryTicket)
}

public enum ModelLibraryEffectID: Hashable, Sendable {
    case resource(String)
    case stopping(String)
    case demo
}

public enum ModelLibraryFeature: Feature {
    public typealias State = ModelLibraryState
    public typealias Action = ModelLibraryAction
    public typealias Effect = ModelLibraryEffect
    public typealias EffectID = ModelLibraryEffectID
    public typealias Operation = Command<Effect, EffectID>

    public static func reduce(_ state: inout State, _ action: Action) -> [Operation] {
        switch action {
        case .refresh:
            return inspect(&state)
        case .queryChanged(let query):
            state.query = query
        case .openDemo(let selection):
            guard state.canBrowse(selection) else {
                state.failure = .unavailable
                return []
            }
            return navigate(&state, to: .demo(selection))
        case .closeDemo:
            return navigate(&state, to: .library)
        case .suspendDemo:
            guard state.pendingDestination == nil, let session = state.session else { return [] }
            state.failure = nil
            return stop(&state, session, then: .demo(session.selection))
        case .selectArtifact(let modelID, let artifactID):
            return select(&state, artifactID: artifactID, for: modelID)
        case .capabilityChanged(let capability, let value):
            state.environment.capabilities[capability] = value
        case .inspected(let artifactID, let request, let resource):
            guard matches(state, artifactID, request, phase: .inspecting) else { return [] }
            complete(&state, artifactID: artifactID, resource: resource)
        case .downloadRequested(let artifactID):
            return download(&state, artifactID: artifactID)
        case .progress(let artifactID, let request, let progress):
            guard let installation = state.installations[artifactID], installation.requestID == request,
                case .downloading(let previous) = installation.phase,
                progress.completedBytes >= (previous?.completedBytes ?? 0)
            else { return [] }
            state.installations[artifactID]?.phase = .downloading(progress)
        case .downloaded(let artifactID, let request, let directory):
            guard let installation = state.installations[artifactID], installation.requestID == request,
                case .downloading = installation.phase, let model = state.catalog.model(containing: artifactID),
                let artifact = model.artifact(artifactID)
            else { return [] }
            guard directory.isFileURL else {
                complete(&state, artifactID: artifactID, failure: .invalidResource)
                return []
            }
            state.installations[artifactID]?.phase = .validating
            return [.run(.validate(model, artifact, request, directory), id: .resource(artifactID))]
        case .validated(let artifactID, let request, let resource):
            guard matches(state, artifactID, request, phase: .validating) else { return [] }
            complete(&state, artifactID: artifactID, resource: resource)
        case .failed(let artifactID, let request, let failure):
            guard let installation = state.installations[artifactID], installation.requestID == request,
                installation.phase.isBusy, installation.phase != .stopping
            else { return [] }
            complete(&state, artifactID: artifactID, failure: failure)
        case .cancelRequested(let artifactID):
            return cancel(&state, artifactID: artifactID)
        case .transferStopped(let artifactID, let request):
            guard matches(state, artifactID, request, phase: .stopping) else { return [] }
            complete(&state, artifactID: artifactID)
        case .deleteRequested(let artifactID):
            return delete(&state, artifactID: artifactID)
        case .deleted(let artifactID, let request):
            guard matches(state, artifactID, request, phase: .deleting) else { return [] }
            complete(&state, artifactID: artifactID)
        case .demoStopped(let sessionID, let request):
            guard state.session?.id == sessionID, state.stopRequestID == request,
                let destination = state.pendingDestination
            else { return [] }
            if let artifactID = state.pendingArtifact, let model = state.catalog.model(containing: artifactID) {
                state.selectedArtifacts[model.id] = artifactID
            }
            state.session = nil
            state.stopRequestID = nil
            state.pendingDestination = nil
            state.pendingArtifact = nil
            state.failure = nil
            apply(&state, destination: destination)
        case .demoStopFailed(let sessionID, let request, let message):
            guard state.session?.id == sessionID, state.stopRequestID == request,
                state.pendingDestination != nil
            else { return [] }
            state.failure = .operation(message)
            state.stopRequestID = nil
        case .retryDemoStop:
            guard let session = state.session, state.pendingDestination != nil, state.stopRequestID == nil
            else { return [] }
            let request = state.allocateID()
            state.stopRequestID = request
            state.failure = nil
            return [.run(.stopDemo(session, request), id: .demo)]
        case .dismissFailure:
            state.failure = nil
        }
        return []
    }

    private static func inspect(_ state: inout State) -> [Operation] {
        let environment = state.environment
        let candidates = state.catalog.sections(platform: environment.platform, audience: environment.audience)
            .flatMap { section in section.artifacts.map { (section.model, $0) } }
        return candidates.compactMap { model, artifact in
            guard let installation = state.installations[artifact.id], !installation.phase.isBusy,
                state.session?.resource.artifactID != artifact.id
            else { return nil }
            let request = state.allocateID()
            state.installations[artifact.id]?.phase = .inspecting
            state.installations[artifact.id]?.requestID = request
            state.installations[artifact.id]?.failure = nil
            return .run(.inspect(model, artifact, request), id: .resource(artifact.id))
        }
    }

    private static func select(_ state: inout State, artifactID: String, for modelID: String) -> [Operation] {
        guard state.pendingDestination == nil else { return [] }
        guard state.artifacts(for: modelID).contains(where: { $0.id == artifactID }) else {
            state.failure = .unavailable
            return []
        }
        guard state.selectedArtifacts[modelID] != artifactID else { return [] }
        state.failure = nil
        if let session = state.session, session.selection.modelID == modelID {
            state.pendingArtifact = artifactID
            return stop(&state, session, then: .demo(session.selection))
        }
        state.selectedArtifacts[modelID] = artifactID
        state.activateIfReady()
        return []
    }

    private static func download(_ state: inout State, artifactID: String) -> [Operation] {
        guard let model = state.catalog.model(containing: artifactID),
            let artifact = state.artifacts(for: model.id).first(where: { $0.id == artifactID }),
            let installation = state.installations[artifactID], !installation.phase.isBusy,
            installation.phase != .ready
        else { return [] }
        guard let download = artifact.download else {
            state.installations[artifactID]?.failure = .downloadNotConfigured
            return []
        }
        guard download.isValid else {
            state.installations[artifactID]?.failure = .invalidDownload
            return []
        }
        let request = state.allocateID()
        state.installations[artifactID]?.phase = .downloading(nil)
        state.installations[artifactID]?.requestID = request
        state.installations[artifactID]?.resource = nil
        state.installations[artifactID]?.failure = nil
        return [.run(.download(model, artifact, request), id: .resource(artifactID))]
    }

    private static func cancel(_ state: inout State, artifactID: String) -> [Operation] {
        guard let installation = state.installations[artifactID], let request = installation.requestID
        else { return [] }
        switch installation.phase {
        case .inspecting, .downloading, .validating:
            state.installations[artifactID]?.phase = .stopping
            return [
                .cancel(.resource(artifactID)),
                .run(.stopTransfer(artifactID: artifactID, request), id: .stopping(artifactID)),
            ]
        case .missing, .stopping, .ready, .deleting, .failed: return []
        }
    }

    private static func delete(_ state: inout State, artifactID: String) -> [Operation] {
        guard state.pendingDestination == nil, let model = state.catalog.model(containing: artifactID),
            state.artifacts(for: model.id).contains(where: { $0.id == artifactID }),
            let installation = state.installations[artifactID], installation.phase == .ready,
            let resource = installation.resource
        else { return [] }
        guard state.session?.selection.modelID != model.id else {
            state.failure = .modelInUse
            return []
        }
        let request = state.allocateID()
        state.installations[artifactID]?.phase = .deleting
        state.installations[artifactID]?.requestID = request
        state.installations[artifactID]?.failure = nil
        return [.run(.delete(resource, request), id: .resource(artifactID))]
    }

    private static func navigate(_ state: inout State, to destination: ModelDestination) -> [Operation] {
        guard state.pendingDestination == nil else { return [] }
        if case .demo(let selected) = destination, state.selection == selected { return [] }
        state.failure = nil
        if let session = state.session { return stop(&state, session, then: destination) }
        apply(&state, destination: destination)
        return []
    }

    private static func stop(
        _ state: inout State, _ session: ModelDemoSession, then destination: ModelDestination
    ) -> [Operation] {
        let request = state.allocateID()
        state.pendingDestination = destination
        state.stopRequestID = request
        return [.run(.stopDemo(session, request), id: .demo)]
    }

    private static func apply(_ state: inout State, destination: ModelDestination) {
        switch destination {
        case .library: state.selection = nil
        case .demo(let selection): state.selection = selection
        }
        state.activateIfReady()
    }

    private static func complete(
        _ state: inout State, artifactID: String, resource: ValidatedModelResource? = nil,
        failure: ModelLibraryFailure? = nil
    ) {
        guard let model = state.catalog.model(containing: artifactID), let artifact = model.artifact(artifactID)
        else { return }
        let rejection: ModelLibraryFailure? =
            if resource == nil { nil } else if artifact.download == nil {
                .downloadNotConfigured
            } else if artifact.download?.isValid != true {
                .invalidDownload
            } else if resource?.matches(model: model, artifact: artifact) != true { .invalidResource } else { nil }
        let failure = rejection ?? failure
        state.installations[artifactID]?.requestID = nil
        state.installations[artifactID]?.resource = failure == nil ? resource : nil
        state.installations[artifactID]?.failure = failure
        state.installations[artifactID]?.phase =
            if failure != nil { .failed } else if resource == nil { .missing } else { .ready }
        state.activateIfReady()
    }

    private static func matches(
        _ state: State, _ artifactID: String, _ request: LibraryTicket, phase: ModelInstallationPhase
    ) -> Bool {
        state.installations[artifactID]?.requestID == request && state.installations[artifactID]?.phase == phase
    }
}
