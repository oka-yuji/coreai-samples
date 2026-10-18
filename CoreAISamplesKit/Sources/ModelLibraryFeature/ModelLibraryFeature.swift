import FeatureStore
import Foundation

public enum ModelLibraryAction: Sendable {
    case refresh
    case queryChanged(String)
    case openDemo(DemoSelection)
    case closeDemo
    case inspected(String, Int, ValidatedModelResource?)
    case downloadRequested(String)
    case progress(String, Int, ModelTransferProgress)
    case downloaded(String, Int, URL)
    case validated(String, Int, ValidatedModelResource)
    case failed(String, Int, ModelLibraryFailure)
    case cancelRequested(String)
    case transferStopped(String, Int)
    case deleteRequested(String)
    case deleted(String, Int)
    case demoStopped(Int, Int)
    case demoStopFailed(Int, Int, String)
    case retryDemoStop
    case dismissFailure
}

public enum ModelLibraryEffect: Equatable, Sendable {
    case inspect(LibraryModel, Int)
    case download(LibraryModel, Int)
    case validate(LibraryModel, Int, URL)
    case stopTransfer(String, Int)
    case delete(ValidatedModelResource, Int)
    case stopDemo(ModelDemoSession, Int)
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
        case .inspected(let modelID, let request, let resource):
            guard matches(state, modelID, request, phase: .inspecting) else { return [] }
            complete(&state, modelID: modelID, resource: resource)
        case .downloadRequested(let modelID):
            return download(&state, modelID: modelID)
        case .progress(let modelID, let request, let progress):
            guard let installation = state.installations[modelID], installation.requestID == request,
                case .downloading(let previous) = installation.phase,
                progress.completedBytes >= (previous?.completedBytes ?? 0)
            else { return [] }
            state.installations[modelID]?.phase = .downloading(progress)
        case .downloaded(let modelID, let request, let directory):
            guard let installation = state.installations[modelID], installation.requestID == request,
                case .downloading = installation.phase, let model = state.catalog.model(modelID)
            else { return [] }
            guard directory.isFileURL else {
                complete(&state, modelID: modelID, failure: .invalidResource)
                return []
            }
            state.installations[modelID]?.phase = .validating
            return [.run(.validate(model, request, directory), id: .resource(modelID))]
        case .validated(let modelID, let request, let resource):
            guard matches(state, modelID, request, phase: .validating) else { return [] }
            complete(&state, modelID: modelID, resource: resource)
        case .failed(let modelID, let request, let failure):
            guard let installation = state.installations[modelID], installation.requestID == request,
                installation.phase.isBusy, installation.phase != .stopping
            else { return [] }
            complete(&state, modelID: modelID, failure: failure)
        case .cancelRequested(let modelID):
            return cancel(&state, modelID: modelID)
        case .transferStopped(let modelID, let request):
            guard matches(state, modelID, request, phase: .stopping) else { return [] }
            complete(&state, modelID: modelID)
        case .deleteRequested(let modelID):
            return delete(&state, modelID: modelID)
        case .deleted(let modelID, let request):
            guard matches(state, modelID, request, phase: .deleting) else { return [] }
            complete(&state, modelID: modelID)
        case .demoStopped(let sessionID, let request):
            guard state.session?.id == sessionID, state.stopRequestID == request,
                let destination = state.pendingDestination
            else { return [] }
            state.session = nil
            state.stopRequestID = nil
            state.pendingDestination = nil
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
        let models = state.catalog.sections(platform: state.platform, audience: state.audience).map(\.model)
        return models.compactMap { model in
            guard let installation = state.installations[model.id], !installation.phase.isBusy,
                state.session?.selection.modelID != model.id
            else { return nil }
            let request = state.allocateID()
            state.installations[model.id]?.phase = .inspecting
            state.installations[model.id]?.requestID = request
            state.installations[model.id]?.failure = nil
            return .run(.inspect(model, request), id: .resource(model.id))
        }
    }

    private static func download(_ state: inout State, modelID: String) -> [Operation] {
        guard isVisible(state, modelID), let model = state.catalog.model(modelID),
            let installation = state.installations[modelID], !installation.phase.isBusy,
            installation.phase != .ready
        else { return [] }
        guard let download = model.artifact.download else {
            state.installations[modelID]?.failure = .downloadNotConfigured
            return []
        }
        guard download.isValid else {
            state.installations[modelID]?.failure = .invalidDownload
            return []
        }
        let request = state.allocateID()
        state.installations[modelID]?.phase = .downloading(nil)
        state.installations[modelID]?.requestID = request
        state.installations[modelID]?.resource = nil
        state.installations[modelID]?.failure = nil
        return [.run(.download(model, request), id: .resource(modelID))]
    }

    private static func cancel(_ state: inout State, modelID: String) -> [Operation] {
        guard let installation = state.installations[modelID], let request = installation.requestID
        else { return [] }
        switch installation.phase {
        case .inspecting, .downloading, .validating:
            state.installations[modelID]?.phase = .stopping
            return [
                .cancel(.resource(modelID)),
                .run(.stopTransfer(modelID, request), id: .stopping(modelID)),
            ]
        case .missing, .stopping, .ready, .deleting, .failed: return []
        }
    }

    private static func delete(_ state: inout State, modelID: String) -> [Operation] {
        guard isVisible(state, modelID), let installation = state.installations[modelID],
            installation.phase == .ready, let resource = installation.resource
        else { return [] }
        guard state.session?.selection.modelID != modelID else {
            state.failure = .modelInUse
            return []
        }
        let request = state.allocateID()
        state.installations[modelID]?.phase = .deleting
        state.installations[modelID]?.requestID = request
        state.installations[modelID]?.failure = nil
        return [.run(.delete(resource, request), id: .resource(modelID))]
    }

    private static func navigate(_ state: inout State, to destination: ModelDestination) -> [Operation] {
        guard state.pendingDestination == nil else { return [] }
        if case .demo(let selected) = destination, state.selection == selected { return [] }
        state.failure = nil
        if let session = state.session {
            let request = state.allocateID()
            state.pendingDestination = destination
            state.stopRequestID = request
            return [.run(.stopDemo(session, request), id: .demo)]
        }
        apply(&state, destination: destination)
        return []
    }

    private static func apply(_ state: inout State, destination: ModelDestination) {
        switch destination {
        case .library: state.selection = nil
        case .demo(let selection): state.selection = selection
        }
        state.activateIfReady()
    }

    private static func complete(
        _ state: inout State, modelID: String, resource: ValidatedModelResource? = nil,
        failure: ModelLibraryFailure? = nil
    ) {
        guard let model = state.catalog.model(modelID) else { return }
        let invalid = resource.map { !$0.matches(model) } ?? false
        let distributionFailure: ModelLibraryFailure? =
            if resource == nil { nil } else if model.artifact.download == nil {
                .downloadNotConfigured
            } else if model.artifact.download?.isValid != true { .invalidDownload } else { nil }
        state.installations[modelID]?.requestID = nil
        state.installations[modelID]?.resource = invalid || distributionFailure != nil ? nil : resource
        state.installations[modelID]?.failure = invalid ? .invalidResource : (distributionFailure ?? failure)
        state.installations[modelID]?.phase =
            if invalid || distributionFailure != nil || failure != nil { .failed } else if resource == nil {
                .missing
            } else { .ready }
        state.activateIfReady()
    }

    private static func matches(_ state: State, _ modelID: String, _ request: Int, phase: ModelInstallationPhase)
        -> Bool
    {
        state.installations[modelID]?.requestID == request && state.installations[modelID]?.phase == phase
    }

    private static func isVisible(_ state: State, _ modelID: String) -> Bool {
        state.catalog.sections(platform: state.platform, audience: state.audience).contains { $0.id == modelID }
    }
}
