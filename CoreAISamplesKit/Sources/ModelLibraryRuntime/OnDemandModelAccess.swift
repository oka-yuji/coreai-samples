import ModelLibraryFeature

nonisolated public struct OnDemandModelAccess: Sendable {
    public let resource: ValidatedModelResource
    public let generation: Int
    public let isCurrent: @MainActor @Sendable () -> Bool
    public let validate: @MainActor @Sendable () async -> Bool
    private let acquireLease: @MainActor @Sendable () async throws -> @MainActor @Sendable () async -> Void

    init(
        resource: ValidatedModelResource, generation: Int, isCurrent: @escaping @MainActor @Sendable () -> Bool,
        validate: @escaping @MainActor @Sendable () async -> Bool,
        acquire: @escaping @MainActor @Sendable () async throws -> @MainActor @Sendable () async -> Void
    ) {
        self.resource = resource
        self.generation = generation
        self.isCurrent = isCurrent
        self.validate = validate
        acquireLease = acquire
    }

    @MainActor public func acquire() async throws -> @MainActor @Sendable () async -> Void {
        try await acquireLease()
    }
}
