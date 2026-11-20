import ModelLibraryFeature

nonisolated public struct OnDemandModelAccess: Sendable {
    public let resource: ValidatedModelResource
    public let generation: Int
    private let acquireLease: @MainActor @Sendable () async throws -> @MainActor @Sendable () async -> Void

    init(
        resource: ValidatedModelResource, generation: Int,
        acquire: @escaping @MainActor @Sendable () async throws -> @MainActor @Sendable () async -> Void
    ) {
        self.resource = resource
        self.generation = generation
        acquireLease = acquire
    }

    @MainActor public func acquire() async throws -> @MainActor @Sendable () async -> Void {
        try await acquireLease()
    }
}
