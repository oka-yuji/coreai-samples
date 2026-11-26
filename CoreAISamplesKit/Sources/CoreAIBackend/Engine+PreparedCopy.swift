import Foundation

extension Engine {
    private struct CachedLoadFailure: LocalizedError {
        let underlying: any Error

        var errorDescription: String? { underlying.localizedDescription }
    }

    static func loadingFunction<Value>(markFailure: Bool, _ load: () throws -> Value) throws -> Value {
        do {
            return try load()
        } catch {
            guard markFailure, !(error is CancellationError) else { throw error }
            try Task.checkCancellation()
            throw CachedLoadFailure(underlying: error)
        }
    }

    static func cachedLoadFailure(_ error: any Error) -> Bool {
        error is CachedLoadFailure
    }

    func warmUpPreparedCopy() async throws {
        let previousStats = stats
        let measured = measuresPreparation
        defer {
            reset()
            stats = previousStats
            measuresPreparation = measured
        }
        measuresPreparation = false
        try Task.checkCancellation()
        if hasPrefillFunction {
            try await readInOneCall([0], usePrefillFunction: true)
            reset()
            try Task.checkCancellation()
        }
        try await readInOneCall([0])
        guard hasUsableLogits else { throw EngineError.missingOutput }
        try Task.checkCancellation()
    }
}
