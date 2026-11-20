import ModelLibraryFeature

nonisolated public struct OnDemandModelAccess: Sendable {
    public let resource: ValidatedModelResource
    public let generation: Int
    public let isCurrent: @MainActor @Sendable () -> Bool
    private let acquireLease: @MainActor @Sendable () async throws -> @MainActor @Sendable () async -> Void

    init(
        resource: ValidatedModelResource, generation: Int, isCurrent: @escaping @MainActor @Sendable () -> Bool,
        acquire: @escaping @MainActor @Sendable () async throws -> @MainActor @Sendable () async -> Void
    ) {
        self.resource = resource
        self.generation = generation
        self.isCurrent = isCurrent
        acquireLease = acquire
    }

    @MainActor public func acquire() async throws -> @MainActor @Sendable () async -> Void {
        try await acquireLease()
    }
}
