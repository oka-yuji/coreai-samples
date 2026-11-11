public struct ObjectPreviewGate: Equatable, Sendable {
    private var minimumGeneration = -1
    private var receivedGeneration = -1
    private var expectedGeneration: Int?
    private var candidates: [ObjectCandidate] = []
    private var validUntil = 0.0
    private var rotation = 90.0
    private var captureGeneration: Int?
    private var captureAngle: Double?

    public init() {}

    public mutating func prepareCapture() {
        captureGeneration = nil
        captureAngle = nil
        expectedGeneration = nil
        candidates = []
        validUntil = 0
    }

    public mutating func updateCaptureAngle(_ value: Double) {
        guard let angle = ObjectCameraOrientation.normalized(value), angle != captureAngle else { return }
        captureAngle = angle
        if let captureGeneration { minimumGeneration = max(minimumGeneration, captureGeneration + 1) }
        expectedGeneration = nil
        candidates = []
        validUntil = 0
    }

    public mutating func begin(_ generation: Int, rotation: Double) {
        guard generation >= 0, generation >= minimumGeneration, generation >= receivedGeneration else { return }
        minimumGeneration = generation
        captureGeneration = generation
        captureAngle = rotation
        expectedGeneration = nil
        candidates = []
        validUntil = 0
    }

    public mutating func expect(_ generation: Int?) {
        expectedGeneration = generation
        if let generation { minimumGeneration = max(minimumGeneration, generation) }
    }

    public mutating func receive(
        _ candidates: [ObjectCandidate], generation: Int, validUntil: Double, rotation: Double = 90
    ) {
        guard generation >= minimumGeneration, generation >= receivedGeneration else { return }
        receivedGeneration = generation
        self.candidates = candidates
        self.validUntil = validUntil
        self.rotation = rotation
    }

    public func visible(at time: Double, rotation: Double? = nil) -> [ObjectCandidate] {
        if let rotation, rotation != self.rotation { return [] }
        guard expectedGeneration == receivedGeneration, time.isFinite, validUntil.isFinite, time <= validUntil else {
            return []
        }
        return candidates
    }

    public func expiration(at time: Double, rotation: Double? = nil) -> Double? {
        visible(at: time, rotation: rotation).isEmpty ? nil : validUntil
    }
}
