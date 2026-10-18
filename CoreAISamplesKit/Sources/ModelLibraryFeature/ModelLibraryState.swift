import Foundation

public struct ValidatedModelResource: Equatable, Sendable {
    public let modelID: String
    public let artifactID: String
    public let contract: String
    public let directory: URL
    public let sizeBytes: Int64

    public init(modelID: String, artifactID: String, contract: String, directory: URL, sizeBytes: Int64) {
        self.modelID = modelID
        self.artifactID = artifactID
        self.contract = contract
        self.directory = directory
        self.sizeBytes = sizeBytes
    }

    public func matches(_ model: LibraryModel) -> Bool {
        modelID == model.id && artifactID == model.artifact.id && contract == model.artifact.contract
            && directory.isFileURL && sizeBytes >= 0
    }
}

public struct ModelTransferProgress: Equatable, Sendable {
    public let completedBytes: Int64
    public let totalBytes: Int64?

    public init?(completedBytes: Int64, totalBytes: Int64?) {
        guard completedBytes >= 0, totalBytes.map({ $0 > 0 && completedBytes <= $0 }) ?? true else { return nil }
        self.completedBytes = completedBytes
        self.totalBytes = totalBytes
    }

    public var fraction: Double? { totalBytes.map { Double(completedBytes) / Double($0) } }
}

public enum ModelInstallationPhase: Equatable, Sendable {
    case missing
    case inspecting
    case downloading(ModelTransferProgress?)
    case validating
    case stopping
    case ready
    case deleting
    case failed

    public var isBusy: Bool {
        switch self {
        case .inspecting, .downloading, .validating, .stopping, .deleting: true
        case .missing, .ready, .failed: false
        }
    }
}

public enum ModelLibraryFailure: Equatable, Sendable {
    case unavailable
    case downloadNotConfigured
    case invalidDownload
    case invalidResource
    case modelInUse
    case operation(String)
}

public struct ModelInstallation: Equatable, Sendable {
    public internal(set) var phase: ModelInstallationPhase = .missing
    public internal(set) var requestID: Int?
    public internal(set) var resource: ValidatedModelResource?
    public internal(set) var failure: ModelLibraryFailure?

    public init() {}
}

public struct ModelDemoSession: Equatable, Sendable {
    public let id: Int
    public let selection: DemoSelection
    public let resource: ValidatedModelResource
}

public enum ModelDestination: Equatable, Sendable {
    case library
    case demo(DemoSelection)
}

public enum DemoAccess: Equatable, Sendable {
    case unavailable
    case preview(ModelInstallationPhase)
    case ready(ModelDemoSession)
    case stopping
}

public struct ModelLibraryState: Equatable, Sendable {
    public let catalog: ModelCatalog
    public let platform: LibraryPlatform
    public let audience: LibraryAudience
    public internal(set) var query = ""
    public internal(set) var installations: [String: ModelInstallation]
    public internal(set) var selection: DemoSelection?
    public internal(set) var session: ModelDemoSession?
    public internal(set) var pendingDestination: ModelDestination?
    public internal(set) var stopRequestID: Int?
    public internal(set) var failure: ModelLibraryFailure?
    var nextID = 0

    public init(catalog: ModelCatalog, platform: LibraryPlatform, audience: LibraryAudience = .release) {
        self.catalog = catalog
        self.platform = platform
        self.audience = audience
        installations = Dictionary(uniqueKeysWithValues: catalog.models.map { ($0.id, ModelInstallation()) })
    }

    public var sections: [ModelSection] { catalog.sections(platform: platform, audience: audience, query: query) }

    public func canBrowse(_ selection: DemoSelection) -> Bool {
        catalog.allows(selection, platform: platform, audience: audience)
    }

    public func access(to selection: DemoSelection) -> DemoAccess {
        guard canBrowse(selection) else { return .unavailable }
        if pendingDestination != nil { return .stopping }
        if let session, session.selection == selection { return .ready(session) }
        return .preview(installations[selection.modelID]?.phase ?? .missing)
    }

    public func permits(_ sessionID: Int, selection: DemoSelection) -> Bool {
        guard case .ready(let session) = access(to: selection), session.id == sessionID,
            let model = catalog.model(selection.modelID), let installed = installations[model.id],
            installed.phase == .ready, installed.resource == session.resource,
            session.resource.matches(model), model.artifact.download?.isValid == true
        else { return false }
        return true
    }

    mutating func allocateID() -> Int {
        nextID += 1
        return nextID
    }

    mutating func activateIfReady() {
        guard session == nil, pendingDestination == nil, let selection, canBrowse(selection),
            let model = catalog.model(selection.modelID), let installed = installations[model.id],
            installed.phase == .ready, let resource = installed.resource, resource.matches(model),
            model.artifact.download?.isValid == true
        else { return }
        session = ModelDemoSession(id: allocateID(), selection: selection, resource: resource)
    }
}
