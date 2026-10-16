public struct ObjectPreviewGate: Equatable, Sendable {
    private var minimumGeneration = -1
    private var receivedGeneration = -1
    private var expectedGeneration: Int?
    private var candidates: [ObjectCandidate] = []
    private var validUntil = 0.0

    public init() {}

    public mutating func expect(_ generation: Int?) {
        expectedGeneration = generation
        if let generation { minimumGeneration = max(minimumGeneration, generation) }
    }

    public mutating func receive(_ candidates: [ObjectCandidate], generation: Int, validUntil: Double) {
        guard generation >= minimumGeneration, generation >= receivedGeneration else { return }
        receivedGeneration = generation
        self.candidates = candidates
        self.validUntil = validUntil
    }

    public func visible(at time: Double) -> [ObjectCandidate] {
        guard expectedGeneration == receivedGeneration, time.isFinite, validUntil.isFinite, time <= validUntil else {
            return []
        }
        return candidates
    }
}
