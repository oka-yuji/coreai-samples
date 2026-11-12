import ChronosCore
import ChronosRuntime
import ForecastFeature
import Foundation
import ModelLibraryFeature

@MainActor
public protocol ForecastService {
    func load(refresh: Bool) async throws -> ForecastDataset
    func predict(_ request: ForecastRequest) async throws -> ForecastResult
    func release() async
}

@MainActor
public final class LiveForecastService: ForecastService {
    private let resource: ValidatedModelResource?
    private let repository: SalesDataRepository
    private let permits: @MainActor () -> Bool
    private let makeRuntime: @MainActor (URL, String) -> ChronosRuntime
    private var dataset: ForecastDataset?
    private var runtime: ChronosRuntime?
    private var busy = false
    private var closing = false
    private var operations: [CheckedContinuation<Void, Never>] = []
    private var releases: [CheckedContinuation<Void, Never>] = []

    public init(
        resource: ValidatedModelResource? = nil, repository: SalesDataRepository = SalesDataRepository(),
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

    public func load(refresh: Bool) async throws -> ForecastDataset {
        try begin()
        defer { finish() }
        let loaded = try await repository.load(refresh: refresh)
        try check()
        dataset = loaded
        return loaded
    }

    public func predict(_ request: ForecastRequest) async throws -> ForecastResult {
        try begin()
        defer { finish() }
        guard let dataset,
            try dataset.request(seriesID: request.seriesID, originIndex: request.originIndex, horizon: request.horizon)
                == request
        else { throw ForecastFailure.invalidRequest }
        guard let resource, resource.modelID == ChronosDemoCatalog.salesSelection.modelID,
            resource.artifactID == "chronos2-fp32", resource.contract == "chronos2-fp32-v1",
            resource.sourceKind == .localManifest, resource.upstreamRevision == ChronosContract.revision
        else { throw ForecastFailure.unavailable }
        let context = try ChronosContext(request.context.map { $0 ?? .nan })
        if runtime == nil { runtime = makeRuntime(resource.directory, resource.revision) }
        guard let runtime else { throw ForecastFailure.unavailable }
        let result = try await runtime.predict(context)
        try check()
        guard result.identity.manifest.sha256 == resource.revision, result.identity.source == .expected,
            result.rawCount == request.context.count, result.paddedCount == context.paddedValues.count
        else { throw ForecastFailure.predictionFailed }
        return try ForecastResult(
            request: request, raw: result.prediction, identity: result.identity,
            seconds: result.seconds, loaded: result.loaded, timingBoundary: result.timingBoundary)
    }

    public func release() async {
        if closing {
            await withCheckedContinuation { releases.append($0) }
            return
        }
        closing = true
        if busy { await withCheckedContinuation { operations.append($0) } }
        let previous = runtime
        runtime = nil
        await previous?.release()
        closing = false
        let waiting = releases
        releases.removeAll()
        for waiter in waiting { waiter.resume() }
    }

    private func begin() throws {
        try check()
        guard !busy else { throw ForecastFailure.busy }
        busy = true
    }

    private func check() throws {
        try Task.checkCancellation()
        guard permits(), !closing else { throw CancellationError() }
    }

    private func finish() {
        busy = false
        let waiting = operations
        operations.removeAll()
        for waiter in waiting { waiter.resume() }
    }
}
