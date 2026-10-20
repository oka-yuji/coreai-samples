import Foundation

@_spi(Verification) public struct GenerationMeasurement: Sendable, Equatable, Encodable {
    public enum Level: String, Sendable, Equatable, Encodable {
        case off
        case summary
        case trace
    }

    public enum Outcome: String, Sendable, Equatable, Encodable {
        case stopToken
        case maxTokens
        case cancelled
        case failed
    }

    public struct Step: Sendable, Equatable, Encodable {
        public var sampleSeconds: Double
        public var textSeconds: Double?
        public var sendSeconds: Double?
        public var stepSeconds: Double?
    }

    public let requestID: UUID
    public let level: Level
    public var outcome: Outcome?
    public var error: String?
    public var promptTokens: Int
    public var prefilledTokens = 0
    public var sampledTokens = 0
    public var steps = 0
    public var sends = 0
    public var reloadSeconds: Double?
    public var readSeconds = 0.0
    public var firstTokenSeconds: Double?
    public var firstSendSeconds: Double?
    public var lastTokenSeconds: Double?
    public var drainSeconds = 0.0
    public var totalSeconds = 0.0
    public var sampleSeconds = 0.0
    public var textSeconds = 0.0
    public var sendSeconds = 0.0
    public var stepSeconds = 0.0
    public var prepareSeconds = 0.0
    public var trace: [Step] = []
}

struct GenerationProbe {
    private(set) var record: GenerationMeasurement
    private let start: ContinuousClock.Instant
    private var lap: ContinuousClock.Instant
    private var preparedBeforeDecode: Double?

    init?(
        _ level: GenerationMeasurement.Level, requestID: UUID, promptTokens: Int, reloadSeconds: Double?,
        start: ContinuousClock.Instant
    ) {
        guard level != .off else { return nil }
        record = GenerationMeasurement(
            requestID: requestID, level: level, promptTokens: promptTokens, reloadSeconds: reloadSeconds)
        self.start = start
        lap = start
    }

    mutating func read(until end: ContinuousClock.Instant, prefilledTokens: Int, prepareSeconds: Double) {
        record.readSeconds = (end - start) / .seconds(1)
        record.prefilledTokens = prefilledTokens
        preparedBeforeDecode = prepareSeconds
        lap = end
    }

    mutating func sampled() {
        let now = ContinuousClock.now
        let seconds = lapSeconds(until: now)
        record.sampleSeconds += seconds
        record.firstTokenSeconds = record.firstTokenSeconds ?? (now - start) / .seconds(1)
        record.lastTokenSeconds = (now - start) / .seconds(1)
        if record.level == .trace { record.trace.append(.init(sampleSeconds: seconds)) }
    }

    mutating func decoded() {
        let seconds = lapSeconds(until: .now)
        record.textSeconds += seconds
        if let last = record.trace.indices.last { record.trace[last].textSeconds = seconds }
    }

    mutating func sent(_ events: Int) {
        let now = ContinuousClock.now
        let seconds = lapSeconds(until: now)
        record.sendSeconds += seconds
        count(events, sentAt: now)
        if let last = record.trace.indices.last { record.trace[last].sendSeconds = seconds }
    }

    mutating func stepped() {
        let seconds = lapSeconds(until: .now)
        record.stepSeconds += seconds
        record.steps += 1
        if let last = record.trace.indices.last { record.trace[last].stepSeconds = seconds }
    }

    mutating func finish(
        _ outcome: GenerationMeasurement.Outcome?, drained events: Int, until end: ContinuousClock.Instant,
        sampledTokens: Int, prepareSeconds: Double
    ) {
        record.drainSeconds = lapSeconds(until: end)
        count(events, sentAt: end)
        close(outcome, at: end, sampledTokens: sampledTokens, prepareSeconds: prepareSeconds)
    }

    mutating func fail(_ error: any Error, sampledTokens: Int, prefilledTokens: Int, prepareSeconds: Double) {
        let now = ContinuousClock.now
        if preparedBeforeDecode == nil {
            record.readSeconds = (now - start) / .seconds(1)
            record.prefilledTokens = prefilledTokens
        }
        record.error = String(describing: error)
        close(
            error is CancellationError ? .cancelled : .failed, at: now, sampledTokens: sampledTokens,
            prepareSeconds: prepareSeconds)
    }

    private mutating func close(
        _ outcome: GenerationMeasurement.Outcome?, at end: ContinuousClock.Instant, sampledTokens: Int,
        prepareSeconds: Double
    ) {
        record.outcome = outcome
        record.totalSeconds = (end - start) / .seconds(1)
        record.sampledTokens = sampledTokens
        record.prepareSeconds = preparedBeforeDecode.map { prepareSeconds - $0 } ?? 0
    }

    private mutating func count(_ events: Int, sentAt time: ContinuousClock.Instant) {
        guard events > 0 else { return }
        record.sends += 1
        record.firstSendSeconds = record.firstSendSeconds ?? (time - start) / .seconds(1)
    }

    private mutating func lapSeconds(until now: ContinuousClock.Instant) -> Double {
        defer { lap = now }
        return (now - lap) / .seconds(1)
    }
}
