import Foundation

public struct ModelArtifactIdentity: Hashable, Sendable {
    public let modelID: String
    public let artifactID: String
    public let contract: String
    public let revision: String
}

public struct ValidatedModelResource: Equatable, Sendable {
    public let modelID: String
    public let artifactID: String
    public let contract: String
    public let revision: String
    public let upstreamRevision: String?
    public let directory: URL
    public let sizeBytes: Int64

    public init(
        modelID: String, artifactID: String, contract: String, revision: String, upstreamRevision: String?,
        directory: URL, sizeBytes: Int64
    ) {
        self.modelID = modelID
        self.artifactID = artifactID
        self.contract = contract
        self.revision = revision
        self.upstreamRevision = upstreamRevision
        self.directory = directory
        self.sizeBytes = sizeBytes
    }

    public var identity: ModelArtifactIdentity {
        ModelArtifactIdentity(modelID: modelID, artifactID: artifactID, contract: contract, revision: revision)
    }

    public func matches(model: LibraryModel, artifact: ModelArtifact) -> Bool {
        modelID == model.id && model.artifact(artifactID) == artifact && contract == artifact.contract
            && revision == artifact.download?.revision && upstreamRevision == artifact.upstream?.revision
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

public struct LibraryTicket: Hashable, Sendable {
    public let owner: UUID
    public let number: Int

    public init(owner: UUID, number: Int) {
        self.owner = owner
        self.number = number
    }
}

public enum CapabilityState: Equatable, Sendable {
    case available, undetermined, denied, unsupported
}

public struct LibraryEnvironment: Equatable, Sendable {
    public let platform: LibraryPlatform
    public let audience: LibraryAudience
    public let connectedDemos: Set<String>
    public var capabilities: [DemoCapability: CapabilityState]

    public init(
        platform: LibraryPlatform, audience: LibraryAudience = .release, connectedDemos: Set<String> = [],
        capabilities: [DemoCapability: CapabilityState] = [:]
    ) {
        self.platform = platform
        self.audience = audience
        self.connectedDemos = connectedDemos
        self.capabilities = capabilities
    }
}

public struct ModelInstallation: Equatable, Sendable {
    public internal(set) var phase: ModelInstallationPhase = .missing
    public internal(set) var requestID: LibraryTicket?
    public internal(set) var resource: ValidatedModelResource?
    public internal(set) var failure: ModelLibraryFailure?

    public init() {}
}

public struct ModelDemoSession: Equatable, Sendable {
    public let id: LibraryTicket
    public let selection: DemoSelection
    public let resource: ValidatedModelResource

    public var identity: ModelArtifactIdentity { resource.identity }
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

public enum DemoPreparation: Equatable, Sendable {
    case pending, preparing, available, occupied, unavailable
}

public struct ModelLibraryState: Equatable, Sendable {
    public let catalog: ModelCatalog
    public let owner: UUID
    public internal(set) var environment: LibraryEnvironment
    public internal(set) var query = ""
    public internal(set) var installations: [String: ModelInstallation]
    public internal(set) var selectedArtifacts: [String: String]
    public internal(set) var selection: DemoSelection?
    public internal(set) var session: ModelDemoSession?
    public internal(set) var demoPreparation: DemoPreparation = .pending
    public internal(set) var pendingDestination: ModelDestination?
    public internal(set) var pendingArtifact: String?
    public internal(set) var pendingLoss: String?
    public internal(set) var stopRequestID: LibraryTicket?
    public internal(set) var failure: ModelLibraryFailure?
    var nextID = 0

    public init(catalog: ModelCatalog, environment: LibraryEnvironment, owner: UUID) {
        self.catalog = catalog
        self.environment = environment
        self.owner = owner
        installations = Dictionary(
            uniqueKeysWithValues: catalog.models.flatMap(\.artifacts).map { ($0.id, ModelInstallation()) })
        selectedArtifacts = Dictionary(uniqueKeysWithValues: catalog.models.map { ($0.id, $0.defaultArtifactID) })
    }

    public var sections: [ModelSection] {
        catalog.sections(platform: environment.platform, audience: environment.audience, query: query)
    }

    public func canBrowse(_ selection: DemoSelection) -> Bool {
        catalog.allows(selection, platform: environment.platform, audience: environment.audience)
    }

    public func artifacts(for modelID: String) -> [ModelArtifact] {
        catalog.sections(platform: environment.platform, audience: environment.audience)
            .first { $0.id == modelID }?.artifacts ?? []
    }

    public func selectedArtifact(for modelID: String) -> ModelArtifact? {
        artifacts(for: modelID).first { $0.id == selectedArtifacts[modelID] }
    }

    public func variantTitle(for modelID: String) -> String? {
        guard let artifact = selectedArtifact(for: modelID),
            artifacts(for: modelID).count > 1 || artifact.download != nil
        else { return nil }
        return artifact.title
    }

    public func rowDetail(for model: LibraryModel, locale: Locale) -> String {
        let size = selectedArtifact(for: model.id).flatMap { installations[$0.id]?.resource?.sizeBytes ?? $0.sizeBytes }
        let formatted = size?.formatted(.byteCount(style: .file).locale(locale))
        return [model.classification, variantTitle(for: model.id), formatted].compactMap { $0 }.joined(separator: " · ")
    }

    public func isConnected(_ selection: DemoSelection) -> Bool {
        environment.connectedDemos.contains(selection.demoID)
    }

    public func access(to selection: DemoSelection) -> DemoAccess {
        guard canBrowse(selection), let artifact = selectedArtifact(for: selection.modelID) else { return .unavailable }
        if pendingDestination != nil { return .stopping }
        if let session, session.selection == selection { return .ready(session) }
        return .preview(installations[artifact.id]?.phase ?? .missing)
    }

    public func permits(_ ticket: LibraryTicket, selection: DemoSelection) -> Bool {
        guard case .ready(let session) = access(to: selection), session.id == ticket,
            executableResource(for: selection) == session.resource,
            let link = catalog.model(selection.modelID)?.demos.first(where: { $0.demoID == selection.demoID })
        else { return false }
        return link.requiredCapabilities.allSatisfy { capability in
            switch environment.capabilities[capability] ?? .undetermined {
            case .available, .undetermined: true
            case .denied, .unsupported: false
            }
        }
    }

    mutating func allocateID() -> LibraryTicket {
        nextID += 1
        return LibraryTicket(owner: owner, number: nextID)
    }

    mutating func activateIfReady() {
        guard session == nil, pendingDestination == nil, let selection,
            let resource = executableResource(for: selection)
        else { return }
        session = ModelDemoSession(id: allocateID(), selection: selection, resource: resource)
        demoPreparation = .pending
    }

    private func executableResource(for selection: DemoSelection) -> ValidatedModelResource? {
        guard canBrowse(selection), isConnected(selection), let model = catalog.model(selection.modelID),
            let artifact = selectedArtifact(for: model.id), artifact.download?.isValid == true,
            let installed = installations[artifact.id], installed.phase == .ready,
            let resource = installed.resource, resource.matches(model: model, artifact: artifact)
        else { return nil }
        return resource
    }
}
