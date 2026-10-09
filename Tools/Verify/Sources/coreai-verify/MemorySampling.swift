import Foundation

func sampleMemory<Model: Sendable>(
    turn: Int, retaining model: Model, started: ContinuousClock.Instant = .now,
    waitUntil: @Sendable (ContinuousClock.Instant) async throws -> Void = {
        try await ContinuousClock().sleep(until: $0)
    },
    capture: @Sendable () -> Memory = Memory.now
) async throws -> MemorySampleRecord {
    defer { withExtendedLifetime(model) {} }
    try await waitUntil(started + .seconds(5))
    let memory = capture()
    let elapsed = seconds(ContinuousClock.now - started)
    return MemorySampleRecord(turn: turn, elapsedSeconds: elapsed, memory: memory)
}

struct MemorySampleRecord: Encodable {
    let record = "memory-sample"
    let runtime = "own"
    let turn: Int
    let waitSeconds = 5
    let elapsedSeconds: Double
    let memory: Memory
}
