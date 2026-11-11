import Foundation

public struct ChronosFileIdentity: Codable, Equatable, Sendable {
    public let bytes: Int64
    public let sha256: String

    public init(bytes: Int64, sha256: String) {
        self.bytes = bytes
        self.sha256 = sha256
    }

    public func validate() throws {
        guard bytes > 0, sha256.utf8.count == 64,
            sha256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
        else { throw ChronosError.invalidBundle("file identity") }
    }
}

public struct ChronosSource: Codable, Equatable, Sendable {
    public let modelId: String
    public let revision: String

    public init(modelId: String, revision: String) {
        self.modelId = modelId
        self.revision = revision
    }

    public static let expected = Self(modelId: ChronosContract.source, revision: ChronosContract.revision)
}

public struct ChronosBundleIdentity: Codable, Equatable, Sendable {
    public let source: ChronosSource
    public let manifest: ChronosFileIdentity
    public let files: [String: ChronosFileIdentity]

    public init(source: ChronosSource, manifest: ChronosFileIdentity, files: [String: ChronosFileIdentity]) {
        self.source = source
        self.manifest = manifest
        self.files = files
    }
}
