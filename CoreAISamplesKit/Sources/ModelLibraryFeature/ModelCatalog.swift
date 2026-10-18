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

public struct ModelDownload: Codable, Equatable, Sendable {
    public let repository: String
    public let revision: String

    public init(repository: String, revision: String) {
        self.repository = repository
        self.revision = revision
    }

    public var isValid: Bool {
        let parts = repository.split(separator: "/", omittingEmptySubsequences: false)
        let characters = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        return parts.count == 2
            && parts.allSatisfy { part in
                !part.isEmpty && part != "." && part != ".."
                    && part.unicodeScalars.allSatisfy(characters.contains)
            }
            && revision.count == 40
            && revision.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
}

public struct ModelArtifact: Codable, Equatable, Sendable {
    public let id: String
    public let contract: String
    public let sizeBytes: Int64?
    public let download: ModelDownload?
    public let visibility: CatalogVisibility

    public init(
        id: String, contract: String, sizeBytes: Int64? = nil,
        download: ModelDownload? = nil, visibility: CatalogVisibility = .published
    ) {
        self.id = id
        self.contract = contract
        self.sizeBytes = sizeBytes
        self.download = download
        self.visibility = visibility
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

    public init(
        demoID: String, platforms: Set<LibraryPlatform>, visibility: CatalogVisibility = .published
    ) {
        self.demoID = demoID
        self.platforms = platforms
        self.visibility = visibility
    }
}

public struct LibraryModel: Codable, Equatable, Identifiable, Sendable {
    public let id: String
    public let name: String
    public let category: String
    public let architecture: String?
    public let searchTerms: [String]
    public let artifact: ModelArtifact
    public let platforms: Set<LibraryPlatform>
    public let visibility: CatalogVisibility
    public let demos: [ModelDemoLink]

    public init(
        id: String, name: String, category: String, architecture: String? = nil,
        searchTerms: [String] = [], artifact: ModelArtifact, platforms: Set<LibraryPlatform>,
        visibility: CatalogVisibility = .published, demos: [ModelDemoLink]
    ) {
        self.id = id
        self.name = name
        self.category = category
        self.architecture = architecture
        self.searchTerms = searchTerms
        self.artifact = artifact
        self.platforms = platforms
        self.visibility = visibility
        self.demos = demos
    }

    public var classification: String {
        [category, architecture].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · ")
    }
}

public struct ModelSection: Equatable, Identifiable, Sendable {
    public let model: LibraryModel
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
    case invalidDemo(String)
    case invalidLink(String, String)
}

public struct ModelCatalog: Equatable, Sendable {
    public let models: [LibraryModel]
    public let demos: [DemoDescriptor]

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
                !model.platforms.isEmpty, !model.artifact.id.isEmpty, !model.artifact.contract.isEmpty,
                model.artifact.sizeBytes.map({ $0 >= 0 }) ?? true
            else { throw ModelCatalogError.invalidModel(model.id) }
            guard modelIDs.insert(model.id).inserted else { throw ModelCatalogError.duplicateModel(model.id) }
            guard artifactIDs.insert(model.artifact.id).inserted else {
                throw ModelCatalogError.duplicateArtifact(model.artifact.id)
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
    public func demo(_ id: String) -> DemoDescriptor? { demos.first { $0.id == id } }

    public func sections(
        platform: LibraryPlatform, audience: LibraryAudience, query: String = ""
    ) -> [ModelSection] {
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return models.compactMap { model in
            guard model.visibility.allows(audience), model.artifact.visibility.allows(audience),
                model.platforms.contains(platform)
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
            return visible.isEmpty ? nil : ModelSection(model: model, demos: visible)
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
