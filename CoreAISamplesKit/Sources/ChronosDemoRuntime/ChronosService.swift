import ChronosCore
import ChronosFeature
import ChronosRuntime
import Foundation
import ModelLibraryFeature

@MainActor
public protocol ChronosService {
    func load() async throws -> ChronosLoadedDemo
    func predict(_ request: ChronosForecastRequest) async throws -> ChronosForecastBatch
    func release() async
}

@MainActor
public final class LiveChronosService: ChronosService {
    private let resource: ValidatedModelResource?
    private let permits: @MainActor () -> Bool
    private let makeRuntime: @MainActor (URL, String) -> ChronosRuntime
    private var repository: ChronosRepository?
    private var runtime: ChronosRuntime?

    public init(
        resource: ValidatedModelResource? = nil, repository: ChronosRepository? = nil,
        permits: @escaping @MainActor () -> Bool = { true },
        makeRuntime: @escaping @MainActor (URL, String) -> ChronosRuntime = {
            ChronosRuntime(bundleURL: $0, expectedManifestSHA256: $1)
        }
    ) {
        self.resource = resource
        self.repository = repository
        self.permits = permits
        self.makeRuntime = makeRuntime
    }

    public func load() async throws -> ChronosLoadedDemo {
        try check()
        if repository == nil { repository = try ChronosRepository.bundled() }
        guard let repository else { throw ChronosDemoFailure.invalidData }
        let catalog = try await repository.load()
        try check()
        return ChronosLoadedDemo(catalog: catalog, liveAvailable: resource != nil)
    }

    public func predict(_ request: ChronosForecastRequest) async throws -> ChronosForecastBatch {
        try check()
        guard let repository else { throw ChronosDemoFailure.invalidData }
        let catalog = try await repository.load()
        try check()
        guard catalog.request(scenarioID: request.scenarioID, step: request.originStep, mode: request.mode) == request
        else { throw ChronosDemoFailure.invalidData }
        if request.mode == .precomputed {
            let batch = try await repository.cached(request)
            try check()
            return batch
        }
        guard let resource, resource.modelID == ChronosDemoCatalog.selection.modelID,
            resource.artifactID == "chronos2-fp32", resource.contract == "chronos2-fp32-v1",
            resource.sourceKind == .localManifest, resource.upstreamRevision == ChronosContract.revision
        else { throw ChronosDemoFailure.unavailable }
        try check()
        if runtime == nil { runtime = makeRuntime(resource.directory, resource.revision) }
        guard let runtime else { throw ChronosDemoFailure.unavailable }
        let start = ContinuousClock.now
        var areas: [ChronosAreaForecast] = []
        var identity: ChronosBundleIdentity?
        for series in request.series {
            try check()
            let result = try await runtime.predict(ChronosContext(series.values))
            try check()
            guard result.identity.manifest.sha256 == resource.revision,
                result.identity.source == .expected,
                result.identity.files["Model.aimodel/main.mlirb"]?.sha256 == ChronosRepository.graphSHA256,
                identity == nil || identity == result.identity
            else { throw ChronosDemoFailure.predictionFailed }
            identity = result.identity
            areas.append(ChronosAreaForecast(areaID: series.areaID, raw: result.prediction))
        }
        try check()
        let duration = start.duration(to: .now).components
        return ChronosForecastBatch(
            scenarioID: request.scenarioID, originStep: request.originStep, mode: .live, areas: areas,
            processingSeconds: Double(duration.seconds) + Double(duration.attoseconds) / 1e18,
            identity: identity)
    }

    public func release() async {
        let prior = runtime
        runtime = nil
        await prior?.release()
    }

    private func check() throws {
        try Task.checkCancellation()
        guard permits() else { throw CancellationError() }
    }
}
