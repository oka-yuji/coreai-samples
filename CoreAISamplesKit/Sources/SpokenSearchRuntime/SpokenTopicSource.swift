import Foundation

public struct SpokenTopicSource: Sendable {
    public typealias Release = @MainActor @Sendable () async -> Void

    public let identity: String
    public let directory: URL
    public let isCurrent: @MainActor @Sendable () -> Bool
    let validate: @MainActor @Sendable () async -> Bool
    let acquire: @MainActor @Sendable () async throws -> Release

    public init(
        identity: String, directory: URL, isCurrent: @escaping @MainActor @Sendable () -> Bool = { true },
        validate: @escaping @MainActor @Sendable () async -> Bool = { true },
        acquire: @escaping @MainActor @Sendable () async throws -> Release
    ) {
        self.identity = identity
        self.directory = directory
        self.isCurrent = isCurrent
        self.validate = validate
        self.acquire = acquire
    }
}
