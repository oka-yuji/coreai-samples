import Foundation

public enum LibraryPlatform: String, Codable, Hashable, Sendable {
    case iOS, macOS
}

public enum LibraryAudience: Equatable, Sendable {
    case release, development
}

public enum CatalogVisibility: String, Codable, Sendable {
    case published, development, hidden

    public func allows(_ audience: LibraryAudience) -> Bool {
        self == .published || (self == .development && audience == .development)
    }
}

public enum DemoCapability: String, Codable, CaseIterable, Sendable {
    case camera, microphone, speechRecognition, network
}

public struct ModelDownload: Codable, Equatable, Sendable {
    public let repository: String
    public let revision: String
    public let root: String

    public init(repository: String, revision: String, root: String = "") {
        self.repository = repository
        self.revision = revision
        self.root = root
    }

    public var isValid: Bool {
        let parts = repository.split(separator: "/", omittingEmptySubsequences: false)
        let characters = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        return parts.count == 2
            && parts.allSatisfy { part in
                !part.isEmpty && part != "." && part != ".."
                    && part.unicodeScalars.allSatisfy(characters.contains)
            }
            && isLowercaseHex(revision, count: 40)
    }
}

public struct ModelUpstream: Codable, Equatable, Sendable {
    public let repository: String
    public let revision: String

    public init(repository: String, revision: String) {
        self.repository = repository
        self.revision = revision
    }
}

public enum ModelSourceKind: String, Codable, Hashable, Sendable {
    case huggingFace
    case localManifest
}

public struct ModelLocalBundle: Codable, Equatable, Sendable {
    public let manifestSHA256: String
    public let profileID: String

    public init(manifestSHA256: String, profileID: String) {
        self.manifestSHA256 = manifestSHA256
        self.profileID = profileID
    }

    public var isValid: Bool {
        isLowercaseHex(manifestSHA256, count: 64) && !profileID.isEmpty
    }
}

public struct ModelArtifactFile: Codable, Equatable, Sendable {
    public let path: String
    public let bytes: Int64
    public let sha256: String

    public init(path: String, bytes: Int64, sha256: String) {
        self.path = path
        self.bytes = bytes
        self.sha256 = sha256
    }
}

public struct ModelArtifact: Codable, Equatable, Sendable {
    public let id: String
    public let title: String
    public let contract: String
    public let platforms: Set<LibraryPlatform>
    public let sizeBytes: Int64?
    public let download: ModelDownload?
    public let localBundle: ModelLocalBundle?
    public let upstream: ModelUpstream?
    public let files: [ModelArtifactFile]
    public let visibility: CatalogVisibility

    public init(
        id: String, title: String, contract: String, platforms: Set<LibraryPlatform>, sizeBytes: Int64? = nil,
        download: ModelDownload? = nil, upstream: ModelUpstream? = nil, files: [ModelArtifactFile] = [],
        visibility: CatalogVisibility = .published, localBundle: ModelLocalBundle? = nil
    ) {
        self.id = id
        self.title = title
        self.contract = contract
        self.platforms = platforms
        self.sizeBytes = sizeBytes
        self.download = download
        self.localBundle = localBundle
        self.upstream = upstream
        self.files = files
        self.visibility = visibility
    }

    public var sourceKind: ModelSourceKind? {
        switch (download, localBundle) {
        case (.some, nil): .huggingFace
        case (nil, .some): .localManifest
        default: nil
        }
    }

    public var sourceRevision: String? {
        switch sourceKind {
        case .huggingFace: download?.revision
        case .localManifest: localBundle?.manifestSHA256
        case nil: nil
        }
    }

    public var isSourceValid: Bool {
        switch sourceKind {
        case .huggingFace: download?.isValid == true
        case .localManifest:
            localBundle?.isValid == true
                && files.contains { $0.path == "bundle.json" && $0.sha256 == localBundle?.manifestSHA256 }
        case nil: false
        }
    }

    public var totalFileBytes: Int64? {
        var total: Int64 = 0
        for file in files {
            let (next, overflow) = total.addingReportingOverflow(file.bytes)
            guard file.bytes >= 0, !overflow else { return nil }
            total = next
        }
        return total
    }

    var isWellFormed: Bool {
        !id.isEmpty && !title.isEmpty && !contract.isEmpty && !platforms.isEmpty && (sizeBytes ?? 0) >= 0
            && (download == nil || localBundle == nil) && (localBundle == nil || isSourceValid)
            && totalFileBytes != nil
            && (download.map { !files.isEmpty && ($0.root.isEmpty || isRelativePath($0.root)) } ?? true)
            && files.allSatisfy { isRelativePath($0.path) && $0.bytes >= 0 && isLowercaseHex($0.sha256, count: 64) }
            && Set(files.map(\.path)).count == files.count
    }
}

public struct DemoExplanation: Codable, Equatable, Sendable {
    public let title: String
    public let text: String

    public init(title: String, text: String) {
        self.title = title
        self.text = text
    }
}

public protocol ModelDemoDefinition: Sendable {
    var id: String { get }
    var title: String { get }
    var systemImage: String { get }
    var summary: String { get }
    var explanation: [DemoExplanation] { get }
    var platforms: Set<LibraryPlatform> { get }
    var visibility: CatalogVisibility { get }
}

public struct DemoDescriptor: ModelDemoDefinition, Codable, Equatable, Identifiable, Sendable {
    public let id: String
    public let title: String
    public let systemImage: String
    public let summary: String
    public let explanation: [DemoExplanation]
    public let platforms: Set<LibraryPlatform>
    public let visibility: CatalogVisibility
    public let searchTerms: [String]?

    public init(
        id: String, title: String, systemImage: String, summary: String,
        explanation: [DemoExplanation] = [], platforms: Set<LibraryPlatform>,
        visibility: CatalogVisibility = .published, searchTerms: [String]? = nil
    ) {
        self.id = id
        self.title = title
        self.systemImage = systemImage
        self.summary = summary
        self.explanation = explanation
        self.platforms = platforms
        self.visibility = visibility
        self.searchTerms = searchTerms
    }
}

public struct ModelDemoLink: Codable, Equatable, Sendable {
    public let demoID: String
    public let platforms: Set<LibraryPlatform>
    public let visibility: CatalogVisibility
    public let requiredCapabilities: Set<DemoCapability>
    public let optionalCapabilities: Set<DemoCapability>

    public init(
        demoID: String, platforms: Set<LibraryPlatform>, visibility: CatalogVisibility = .published,
        requiredCapabilities: Set<DemoCapability> = [], optionalCapabilities: Set<DemoCapability> = []
    ) {
        self.demoID = demoID
        self.platforms = platforms
        self.visibility = visibility
        self.requiredCapabilities = requiredCapabilities
        self.optionalCapabilities = optionalCapabilities
    }
}

public struct LibraryModel: Codable, Equatable, Identifiable, Sendable {
    public let id: String
    public let name: String
    public let category: String
    public let architecture: String?
    public let searchTerms: [String]
    public let artifacts: [ModelArtifact]
    public let defaultArtifactID: String
    public let platforms: Set<LibraryPlatform>
    public let visibility: CatalogVisibility
    public let demos: [ModelDemoLink]

    public init(
        id: String, name: String, category: String, architecture: String? = nil,
        searchTerms: [String] = [], artifacts: [ModelArtifact], defaultArtifactID: String,
        platforms: Set<LibraryPlatform>, visibility: CatalogVisibility = .published, demos: [ModelDemoLink]
    ) {
        self.id = id
        self.name = name
        self.category = category
        self.architecture = architecture
        self.searchTerms = searchTerms
        self.artifacts = artifacts
        self.defaultArtifactID = defaultArtifactID
        self.platforms = platforms
        self.visibility = visibility
        self.demos = demos
    }

    public var classification: String {
        [category, architecture].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · ")
    }

    public func artifact(_ id: String) -> ModelArtifact? { artifacts.first { $0.id == id } }
}

public struct ModelSection: Equatable, Identifiable, Sendable {
    public let model: LibraryModel
    public let artifacts: [ModelArtifact]
    public let demos: [DemoDescriptor]
    public var id: String { model.id }
}

public struct DemoSelection: Codable, Equatable, Hashable, Sendable {
    public let modelID: String
    public let demoID: String

    public init(modelID: String, demoID: String) {
        self.modelID = modelID
        self.demoID = demoID
    }
}

public enum ModelCatalogError: Error, Equatable {
    case duplicateModel(String)
    case duplicateArtifact(String)
    case duplicateDemo(String)
    case invalidModel(String)
    case invalidArtifact(String)
    case invalidDemo(String)
    case invalidLink(String, String)
}

public struct ModelCatalog: Equatable, Sendable {
    public let models: [LibraryModel]
    public let demos: [DemoDescriptor]

    public func validateConnections(adapterIDs: [String], unconnectedDemoIDs: Set<String> = []) throws {
        var connected: Set<String> = []
        let defined = Set(demos.map(\.id))
        for id in adapterIDs {
            guard defined.contains(id) else { throw ModelCatalogError.invalidDemo(id) }
            guard connected.insert(id).inserted else { throw ModelCatalogError.duplicateDemo(id) }
        }
        if let invalid = unconnectedDemoIDs.subtracting(defined).sorted().first {
            throw ModelCatalogError.invalidDemo(invalid)
        }
        if let duplicate = connected.intersection(unconnectedDemoIDs).sorted().first {
            throw ModelCatalogError.duplicateDemo(duplicate)
        }
        if let missing = defined.subtracting(connected.union(unconnectedDemoIDs)).sorted().first {
            throw ModelCatalogError.invalidDemo(missing)
        }
    }

    public init(models: [LibraryModel], demos: [DemoDescriptor]) throws {
        var modelIDs: Set<String> = []
        var artifactIDs: Set<String> = []
        var demoIDs: Set<String> = []
        for demo in demos {
            guard !demo.id.isEmpty, !demo.title.isEmpty, !demo.systemImage.isEmpty,
                !demo.summary.isEmpty, !demo.platforms.isEmpty
            else { throw ModelCatalogError.invalidDemo(demo.id) }
            guard demoIDs.insert(demo.id).inserted else { throw ModelCatalogError.duplicateDemo(demo.id) }
        }
        for model in models {
            guard !model.id.isEmpty, !model.name.isEmpty, !model.category.isEmpty,
                !model.platforms.isEmpty, model.artifact(model.defaultArtifactID) != nil
            else { throw ModelCatalogError.invalidModel(model.id) }
            guard modelIDs.insert(model.id).inserted else { throw ModelCatalogError.duplicateModel(model.id) }
            for artifact in model.artifacts {
                guard artifact.isWellFormed else { throw ModelCatalogError.invalidArtifact(artifact.id) }
                guard artifactIDs.insert(artifact.id).inserted else {
                    throw ModelCatalogError.duplicateArtifact(artifact.id)
                }
            }
            var links: Set<String> = []
            for link in model.demos {
                guard demoIDs.contains(link.demoID), links.insert(link.demoID).inserted,
                    !link.platforms.isEmpty
                else { throw ModelCatalogError.invalidLink(model.id, link.demoID) }
            }
        }
        self.models = models
        self.demos = demos
    }

    public func model(_ id: String) -> LibraryModel? { models.first { $0.id == id } }
    public func model(containing artifactID: String) -> LibraryModel? {
        models.first { $0.artifact(artifactID) != nil }
    }
    public func demo(_ id: String) -> DemoDescriptor? { demos.first { $0.id == id } }

    public func sections(
        platform: LibraryPlatform, audience: LibraryAudience, query: String = ""
    ) -> [ModelSection] {
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return models.compactMap { model in
            let artifacts = model.artifacts.filter { $0.visibility.allows(audience) && $0.platforms.contains(platform) }
            guard model.visibility.allows(audience), model.platforms.contains(platform), !artifacts.isEmpty
            else { return nil }
            let modelMatches = ([model.name, model.category, model.architecture ?? ""] + model.searchTerms)
                .contains { $0.lowercased().contains(query) }
            let visible = model.demos.compactMap { link -> DemoDescriptor? in
                guard link.visibility.allows(audience), link.platforms.contains(platform),
                    let demo = demo(link.demoID), demo.visibility.allows(audience), demo.platforms.contains(platform)
                else { return nil }
                guard
                    query.isEmpty || modelMatches
                        || ([demo.title, demo.summary] + (demo.searchTerms ?? []))
                            .contains(where: { $0.lowercased().contains(query) })
                else { return nil }
                return demo
            }
            return visible.isEmpty ? nil : ModelSection(model: model, artifacts: artifacts, demos: visible)
        }
    }

    public func allows(
        _ selection: DemoSelection, platform: LibraryPlatform, audience: LibraryAudience
    ) -> Bool {
        sections(platform: platform, audience: audience).contains { section in
            section.id == selection.modelID && section.demos.contains { $0.id == selection.demoID }
        }
    }
}

private func isRelativePath(_ path: String) -> Bool {
    path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy {
        !$0.isEmpty && $0 != "." && $0 != ".."
    }
}

private func isLowercaseHex(_ value: String, count: Int) -> Bool {
    value.utf8.count == count && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
}
