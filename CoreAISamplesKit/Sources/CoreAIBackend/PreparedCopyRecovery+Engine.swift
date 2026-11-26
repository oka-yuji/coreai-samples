extension PreparedCopyRecovery {
    func loadEngine(
        bundle: ModelBundle, contextLimit: Int,
        makeEngine: (@Sendable (ModelBundle, Int) async throws -> EngineBox)?,
        delete: @Sendable (ModelBundle) throws -> Void, onRecovery: @Sendable () -> Void
    ) async throws -> PreparedCopyLoad<EngineBox> {
        let loader =
            makeEngine ?? {
                EngineBox(try await Engine(bundle: $0, contextLimit: $1, markCachedLoadFailure: true))
            }
        var repairing = false
        do {
            return try await load(
                shouldRetry: Engine.cachedLoadFailure,
                onRecovery: {
                    repairing = true
                    onRecovery()
                },
                delete: { try delete(bundle) }, load: { try await loader(bundle, contextLimit) },
                warmUp: { try await $0.engine.warmUpPreparedCopy() })
        } catch {
            if repairing, !(error is CancellationError) {
                throw PreparedCopyError.recoveryFailed
            }
            throw error
        }
    }
}
