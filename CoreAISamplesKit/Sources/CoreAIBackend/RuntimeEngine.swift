protocol RuntimeEngine: AnyObject {
    var tokens: [Int32] { get }
    var stats: Engine.Stats { get }
    var loadStats: Engine.LoadStats { get }
    var measuresPreparation: Bool { get set }
    var isHybrid: Bool { get }
    var hasLogits: Bool { get }
    func logits() -> [Float16]
    func withLogits<Result>(_ body: (UnsafeBufferPointer<Float16>) -> Result) -> Result
    func checkpoint() -> Engine.Checkpoint
    func restore(_ checkpoint: Engine.Checkpoint) throws
    func reset()
    func unload()
    func warmUpPreparedCopy() async throws
    func prefill(_ input: [Int32], usePrefillFunction: Bool) async throws
    func prefill(
        _ input: [Int32], usePrefillFunction: Bool, onProgress: @Sendable (Int) -> Void
    ) async throws
    func step(_ token: Int32) async throws
}

extension RuntimeEngine {
    var hasUsableLogits: Bool {
        hasLogits && withLogits { $0.contains(where: \.isFinite) }
    }

    func prefill(
        _ input: [Int32], usePrefillFunction: Bool, onProgress: @Sendable (Int) -> Void
    ) async throws {
        onProgress(0)
        try await prefill(input, usePrefillFunction: usePrefillFunction)
        onProgress(input.count)
    }
}

extension Engine: RuntimeEngine {}
