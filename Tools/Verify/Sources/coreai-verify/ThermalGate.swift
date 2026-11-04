import Foundation

struct ThermalStart: Equatable {
    let waitedSeconds: Double
    let state: ProcessInfo.ThermalState

    var stateName: String {
        switch state {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }
}

struct ThermalGate {
    var limit: Duration
    var interval: Duration = .seconds(1)
    var state: () -> ProcessInfo.ThermalState = { ProcessInfo.processInfo.thermalState }
    var now: () -> ContinuousClock.Instant = { ContinuousClock.now }
    var sleep: (Duration) async throws -> Void = { try await Task.sleep(for: $0) }

    func wait() async throws -> ThermalStart {
        let start = now()
        var current = state()
        while current != .nominal, now() - start < limit {
            try await sleep(interval)
            current = state()
        }
        return ThermalStart(waitedSeconds: (now() - start) / .seconds(1), state: current)
    }
}

extension TypedDecideRecord {
    mutating func setStart(_ thermal: ThermalStart, at started: Date) {
        self.started = started
        thermalWaitSeconds = thermal.waitedSeconds
        thermalStateAtStart = thermal.stateName
    }
}
