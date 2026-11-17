public struct VoxCPMEstimatorInput: Sendable {
    public let x: [Float]
    public let mu: [Float]
    public let t: [Float]
    public let cond: [Float]
    public let dt: [Float]
}
