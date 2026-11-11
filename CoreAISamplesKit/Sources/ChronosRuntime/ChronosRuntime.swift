import ChronosCore
import Foundation

public protocol ChronosPredictor: Sendable {
    var identity: ChronosBundleIdentity { get }
    func predict(_ context: ChronosContext) async throws -> ChronosPrediction
}

public struct ChronosResult: Encodable, Sendable {
    public let identity: ChronosBundleIdentity
    public let prediction: ChronosPrediction
    public let rawCount: Int
    public let paddedCount: Int
    public let seconds: Double
    public let loaded: Bool
    public let timingBoundary: String
}

public actor ChronosRuntime {
    enum Event: Sendable, Equatable {
        case releaseWaiting
        case released
    }

    private let factory: @Sendable () async throws -> any ChronosPredictor
    private let observe: @Sendable (Event) -> Void
    private var predictor: (any ChronosPredictor)?
    private var busy = false
    private var closing = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init(bundleURL: URL, expectedManifestSHA256: String? = nil) {
        factory = {
            try await ChronosNativePredictor.load(bundleURL: bundleURL, expectedManifestSHA256: expectedManifestSHA256)
        }
        observe = { _ in }
    }

    public init(factory: @escaping @Sendable () async throws -> any ChronosPredictor) {
        self.factory = factory
        observe = { _ in }
    }

    init(
        factory: @escaping @Sendable () async throws -> any ChronosPredictor,
        observe: @escaping @Sendable (Event) -> Void
    ) {
        self.factory = factory
        self.observe = observe
    }

    public func predict(_ context: ChronosContext) async throws -> ChronosResult {
        guard !busy, !closing else { throw ChronosError.busy }
        try Task.checkCancellation()
        busy = true
        do {
            let result = try await perform(context)
            finish(discard: false)
            return result
        } catch {
            finish(discard: true)
            throw error
        }
    }

    public func release() async {
        guard busy else {
            predictor = nil
            observe(.released)
            return
        }
        closing = true
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
            observe(.releaseWaiting)
        }
    }

    private func perform(_ context: ChronosContext) async throws -> ChronosResult {
        let start = ContinuousClock.now
        let loaded = predictor == nil
        if predictor == nil {
            let created = try await factory()
            try checkActive()
            predictor = created
        }
        guard let predictor else { throw ChronosError.invalidModel }
        let prediction = try await predictor.predict(context)
        try checkActive()
        let duration = start.duration(to: .now).components
        return ChronosResult(
            identity: predictor.identity, prediction: prediction, rawCount: context.rawCount,
            paddedCount: context.paddedValues.count,
            seconds: Double(duration.seconds) + Double(duration.attoseconds) / 1e18, loaded: loaded,
            timingBoundary:
                "Runtime request through validated output; includes bundle hashing, verified snapshot copying and model loading when loaded is true."
        )
    }

    private func checkActive() throws {
        try Task.checkCancellation()
        guard !closing else { throw CancellationError() }
    }

    private func finish(discard: Bool) {
        if discard || closing { predictor = nil }
        busy = false
        guard closing else { return }
        closing = false
        let completions = waiters
        waiters.removeAll()
        observe(.released)
        for completion in completions { completion.resume() }
    }
}
