import Foundation

struct ResidentMaximum: Encodable, Equatable {
    var rssBytes: UInt64 = 0
    var footprintBytes: UInt64 = 0
    var samples = 0
    var failedSamples = 0
    var intervalSeconds = 0.1

    mutating func add(_ memory: Memory) {
        guard let rss = memory.rssBytes, let footprint = memory.footprintBytes else {
            failedSamples += 1
            return
        }
        rssBytes = max(rssBytes, rss)
        footprintBytes = max(footprintBytes, footprint)
        samples += 1
    }
}

func sampleResident<Value>(
    every interval: Duration = .milliseconds(100), started: ContinuousClock.Instant = .now,
    waitUntil: @escaping @Sendable (ContinuousClock.Instant) async throws -> Void = {
        try await ContinuousClock().sleep(until: $0)
    },
    capture: @escaping @Sendable () -> Memory = Memory.now,
    _ operation: () async throws -> Value
) async throws -> (value: Value, resident: ResidentMaximum) {
    try await withThrowingTaskGroup(of: ResidentMaximum.self) { group in
        group.addTask {
            var maximum = ResidentMaximum(intervalSeconds: interval / .seconds(1))
            var deadline = started
            while !Task.isCancelled {
                maximum.add(capture())
                deadline += interval
                do { try await waitUntil(deadline) } catch { break }
            }
            return maximum
        }
        let value = try await operation()
        group.cancelAll()
        var maximum = try await group.next() ?? ResidentMaximum()
        maximum.add(capture())
        return (value, maximum)
    }
}
