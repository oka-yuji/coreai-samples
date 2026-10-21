import Foundation
import ImageGenerationCore
import ModelLibraryFeature

@MainActor
public protocol ImageGenerationService {
    func generate(
        _ request: ImageGenerationRequest, trace: (any ImageTraceSink)?,
        progress: @MainActor @Sendable (ImageExecutionProgress) -> Void
    ) async throws -> ImageGenerationResult
    func stop() async throws
    func release() async throws
    func latestReport() async -> ImageExecutionReport?
}

@MainActor
public final class LiveImageGenerationService: ImageGenerationService {
    public let session: ModelDemoSession
    private let pipeline: ImageRuntimePipeline
    private let permission: @MainActor @Sendable () -> Bool
    private var running = false
    private var delivery: UUID?
    private var released = false
    private var stops = 0
    private var completions: [CheckedContinuation<Void, Never>] = []

    public convenience init(
        session: ModelDemoSession, budget: ImageMemoryBudget,
        isPermitted: @escaping @MainActor @Sendable () -> Bool
    ) {
        self.init(
            session: session,
            pipeline: ImageRuntimePipeline(
                resource: session.resource, budget: budget, backend: CoreAIImageBackend(),
                permission: isPermitted, memory: ImageMemorySampler.capture),
            isPermitted: isPermitted)
    }

    init(
        session: ModelDemoSession, pipeline: ImageRuntimePipeline,
        isPermitted: @escaping @MainActor @Sendable () -> Bool
    ) {
        self.session = session
        self.pipeline = pipeline
        permission = isPermitted
    }

    public func generate(
        _ request: ImageGenerationRequest, trace: (any ImageTraceSink)? = nil,
        progress: @MainActor @Sendable (ImageExecutionProgress) -> Void
    ) async throws -> ImageGenerationResult {
        guard !released else { throw ImageExecutionError.released }
        guard !running, stops == 0 else { throw ImageExecutionError.busy }
        guard permission() else { throw ImageExecutionError.permissionDenied }
        try Task.checkCancellation()
        running = true
        delivery = request.id
        defer {
            running = false
            delivery = nil
            let pending = completions
            completions.removeAll()
            for completion in pending { completion.resume() }
        }
        let requestID = request.id
        let result = try await pipeline.generate(
            request, trace: trace,
            isCurrent: { [weak self] in
                guard let self else { return false }
                return self.delivery == requestID && self.stops == 0 && !self.released
            },
            progress: { [weak self] value in
                guard let self, self.delivery == value.requestID, !self.released, self.permission() else { return }
                progress(value)
            })
        try Task.checkCancellation()
        guard delivery == request.id, !released else { throw CancellationError() }
        guard permission() else { throw ImageExecutionError.permissionDenied }
        return result
    }

    public func stop() async throws {
        delivery = nil
        try await stopAndJoin(releasing: false)
    }

    public func release() async throws {
        released = true
        delivery = nil
        try await stopAndJoin(releasing: true)
    }

    public func latestReport() async -> ImageExecutionReport? { await pipeline.report }

    private func stopAndJoin(releasing: Bool) async throws {
        stops += 1
        defer { stops -= 1 }
        var failure: (any Error)?
        do { try await pipeline.stop(releasing: releasing) } catch { failure = error }
        if running { await withCheckedContinuation { completions.append($0) } }
        if let failure { throw failure }
    }
}
