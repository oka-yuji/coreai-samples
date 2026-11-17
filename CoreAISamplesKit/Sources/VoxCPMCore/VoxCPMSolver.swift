import Foundation

public struct VoxCPMSolver: Sendable {
    public static let timeSpan: [Float] = (0...stepCount).map(time(at:))
    public static let stepCount = 10
    public static let patchSize = 256
    public static let hiddenSize = 2048
    public static let guidanceScale: Float = 2
    public static let swaySamplingCoefficient: Float = 1
    public static let usesZeroStar = true
    public static let zeroInitSteps = 1
    public static let estimatorCallsPerPatch = stepCount - zeroInitSteps

    private static let normEpsilon: Float = 1e-8

    public private(set) var latent: [Float]
    private let mu: [Float]
    private let cond: [Float]
    private var stepIndex = zeroInitSteps

    public init(noise: [Float], ditHidden: [Float], condition: [Float]) throws {
        guard noise.count == Self.patchSize, ditHidden.count == Self.hiddenSize, condition.count == Self.patchSize
        else { throw VoxCPMError.invalidSize }
        latent = noise
        mu = ditHidden + [Float](repeating: 0, count: Self.hiddenSize)
        cond = condition + condition
    }

    public var estimatorInput: VoxCPMEstimatorInput? {
        guard stepIndex < Self.stepCount else { return nil }
        let time = Self.timeSpan[stepIndex]
        return VoxCPMEstimatorInput(x: latent + latent, mu: mu, t: [time, time], cond: cond, dt: [0, 0])
    }

    public mutating func advance(velocity: [Float]) throws {
        guard stepIndex < Self.stepCount else { throw VoxCPMError.solverFinished }
        guard velocity.count == 2 * Self.patchSize else { throw VoxCPMError.invalidSize }
        let conditional = velocity[..<Self.patchSize]
        let unconditional = velocity[Self.patchSize...]
        let scale =
            Self.usesZeroStar
            ? Self.dot(conditional, unconditional) / (Self.dot(unconditional, unconditional) + Self.normEpsilon) : 1
        let stepSize = Self.timeSpan[stepIndex] - Self.timeSpan[stepIndex + 1]
        for index in latent.indices {
            let scaled = velocity[Self.patchSize + index] * scale
            let guided = scaled + Self.guidanceScale * (velocity[index] - scaled)
            latent[index] -= stepSize * guided
        }
        stepIndex += 1
    }

    private static func time(at index: Int) -> Float {
        let start: Float = 1
        let end: Float = 0
        let increment = (end - start) / Float(stepCount)
        let linear =
            index < (stepCount + 1) / 2
            ? start + increment * Float(index) : end - increment * Float(stepCount - index)
        return linear + swaySamplingCoefficient * ((cos(Float(Double.pi / 2) * linear) - 1) + linear)
    }

    private static func dot(_ left: ArraySlice<Float>, _ right: ArraySlice<Float>) -> Float {
        Float(zip(left, right).reduce(0) { $0 + Double($1.0) * Double($1.1) })
    }
}
