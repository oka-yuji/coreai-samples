import Testing

@testable import CoreAIBackend

@Suite struct ModelResolutionTests {
    private struct Failure: Error, Equatable, CustomStringConvertible {
        let description: String
    }

    @Test("When the cache query throws and specializing again also fails, an error with both causes is thrown")
    func queryAndSpecializationFailuresAreBothReported() async {
        let error = await #expect(throws: EngineError.self) {
            try await Engine.resolveModel(
                query: { () throws -> Int? in throw Failure(description: "Missing hash file") },
                fallback: { throw Failure(description: "Specialization failed") })
        }
        #expect(error == .cacheQueryFailed(query: "Missing hash file", specialize: "Specialization failed"))
        for text in [error?.localizedDescription ?? "", error.map { "\($0)" } ?? ""] {
            #expect(text.contains("Missing hash file") && text.contains("Specialization failed"), "\(text)")
        }
    }

    @Test(
        "When the cache query throws and specializing again succeeds, the load stats keep the query failure and cached is false"
    )
    func queryFailureIsKeptAfterSpecialization() async throws {
        let resolved = try await Engine.resolveModel(
            query: { () throws -> Int? in throw Failure(description: "Missing hash file") }, fallback: { 7 })
        #expect(resolved.model == 7)
        #expect(!resolved.cached)
        #expect(resolved.cacheQueryError == "Missing hash file")
    }

    @Test("When the cache query returns nil, only the specialization runs as before and the load stats keep no failure")
    func missRunsOnlyTheSpecialization() async throws {
        var specializations = 0
        let resolved = try await Engine.resolveModel(
            query: { nil },
            fallback: {
                specializations += 1
                return 7
            })
        #expect(specializations == 1)
        #expect(resolved.model == 7)
        #expect(!resolved.cached)
        #expect(resolved.cacheQueryError == nil)
    }

    @Test("When the cache query returns a model, the specialization does not run and cached is true")
    func hitSkipsTheSpecialization() async throws {
        var specializations = 0
        let resolved = try await Engine.resolveModel(
            query: { 3 },
            fallback: {
                specializations += 1
                return 7
            })
        #expect(specializations == 0)
        #expect(resolved.model == 3)
        #expect(resolved.cached)
        #expect(resolved.cacheQueryError == nil)
        #expect(resolved.specializeSeconds == 0)
    }

    @Test(
        "When the cache query throws and the specialization throws CancellationError, the cancellation propagates instead of cacheQueryFailed"
    )
    func cancellationAfterQueryFailureIsNotWrapped() async {
        await #expect(throws: CancellationError.self) {
            try await Engine.resolveModel(
                query: { () throws -> Int? in throw Failure(description: "Missing hash file") },
                fallback: { throw CancellationError() })
        }
    }

    @Test("When the cache query returns nil and the specialization fails, the original error is thrown unchanged")
    func missRethrowsTheSpecializationError() async {
        await #expect(throws: Failure(description: "Specialization failed")) {
            try await Engine.resolveModel(
                query: { () throws -> Int? in nil }, fallback: { throw Failure(description: "Specialization failed") })
        }
    }

    @Test("Only an opted-in cached function load error qualifies for an immediate retry")
    func onlyCachedFunctionLoadErrorsCanRetry() throws {
        let error = Failure(description: "Could not load the function")
        let marked = try #require(
            #expect(throws: (any Error).self) {
                try Engine.loadingFunction(markFailure: true) { throw error }
            })
        #expect(Engine.cachedLoadFailure(marked))
        #expect(!Engine.cachedLoadFailure(error))
        #expect(throws: error) {
            try Engine.loadingFunction(markFailure: false) { throw error }
        }
        #expect(throws: CancellationError.self) {
            try Engine.loadingFunction(markFailure: true) { throw CancellationError() }
        }
        let missing: Int? = try Engine.loadingFunction(markFailure: true) { nil }
        #expect(missing == nil)
    }
}
