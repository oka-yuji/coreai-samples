import Foundation

public enum ChronosContract {
    public static let source = "amazon/chronos-2"
    public static let revision = "29ec3766d36d6f73f0696f85560a422f50e8498c"
    public static let format = "coreai-samples-chronos2/1"
    public static let patchSize = 16
    public static let maximumContext = 8192
    public static let outputShape = [1, 21, 16]
    public static let quantileLevels: [Double] = [
        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
    ]
}

public enum ChronosError: Error, Equatable, Sendable {
    case invalidContext
    case invalidPrediction
    case invalidBundle(String)
    case invalidReference
    case invalidArguments
    case invalidInput
    case outputExists
    case busy
    case invalidModel
    case verificationFailed
}

public struct ChronosContext: Sendable {
    public let rawCount: Int
    public let paddedValues: [Float]

    public init(_ values: [Float]) throws {
        guard (1...ChronosContract.maximumContext).contains(values.count),
            values.contains(where: \.isFinite), !values.contains(where: \.isInfinite)
        else { throw ChronosError.invalidContext }
        rawCount = values.count
        let count =
            ((values.count + ChronosContract.patchSize - 1) / ChronosContract.patchSize)
            * ChronosContract.patchSize
        paddedValues = Array(repeating: Float.nan, count: count - values.count) + values
    }
}

public struct ChronosPrediction: Sendable, Equatable, Codable {
    public let quantiles: [[Float]]

    public init(quantiles: [[Float]]) throws {
        guard quantiles.count == ChronosContract.quantileLevels.count,
            quantiles.allSatisfy({ $0.count == ChronosContract.patchSize && $0.allSatisfy(\.isFinite) })
        else { throw ChronosError.invalidPrediction }
        self.quantiles = quantiles
    }

    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(quantiles: values.decode([[Float]].self, forKey: .quantiles))
    }
}
