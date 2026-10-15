import Foundation

public struct InspectionMeasurement: Equatable, Sendable {
    public let warmupMilliseconds: Double
    public let samples: [Double]
    public let medianMilliseconds: Double
    public let p95Milliseconds: Double

    public init?(warmupMilliseconds: Double, samples: [Double]) {
        guard warmupMilliseconds.isFinite, warmupMilliseconds >= 0, samples.count == 20,
            samples.allSatisfy({ $0.isFinite && $0 >= 0 })
        else { return nil }
        self.warmupMilliseconds = warmupMilliseconds
        self.samples = samples
        let sorted = samples.sorted()
        medianMilliseconds = sorted[9] / 2 + sorted[10] / 2
        p95Milliseconds = sorted[18]
    }
}
