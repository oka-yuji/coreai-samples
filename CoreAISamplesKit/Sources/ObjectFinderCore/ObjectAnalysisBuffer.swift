public struct ObjectAnalysisBuffer<Frame: Sendable>: Sendable {
    public private(set) var latest: Frame?
    private var latestTime = -Double.infinity
    private var deliveredTime = -Double.infinity
    private var requested = false

    public init() {}

    public mutating func offer(_ frame: Frame, at time: Double) -> Frame? {
        guard time.isFinite else { return nil }
        latest = frame
        latestTime = time
        return deliver()
    }

    public mutating func request() -> Frame? {
        requested = true
        return deliver()
    }

    public mutating func reset() {
        self = Self()
    }

    private mutating func deliver() -> Frame? {
        guard requested, let latest,
            latestTime - deliveredTime >= 0.15 || latestTime < deliveredTime
        else { return nil }
        requested = false
        deliveredTime = latestTime
        return latest
    }
}
