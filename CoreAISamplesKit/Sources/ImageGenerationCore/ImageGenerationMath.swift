import Foundation

public enum ImageGenerationError: Error, Equatable, Sendable {
    case invalidProfile(String)
    case invalidShape
    case invalidTokens
    case promptTooLong(Int)
    case inputTooLarge(Int)
    case unsupportedImage
    case nonFinite
    case invalidSeed
}

public enum ImageGenerationPrecision: String, Codable, Sendable {
    case float32, float16, bfloat16

    public func rounded(_ value: Float) throws -> Float {
        guard value.isFinite else { throw ImageGenerationError.nonFinite }
        let result: Float
        switch self {
        case .float32: result = value
        case .float16: result = Float(Float16(value))
        case .bfloat16:
            let bits = value.bitPattern
            result = Float(bitPattern: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) & 0xFFFF_0000)
        }
        guard result.isFinite else { throw ImageGenerationError.nonFinite }
        return result
    }
}

public enum ImageEuler {
    public static let sigmas: [Float] = [1, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
    public static let latentCount = 1_024 * 64

    public static func initialLatents(noise: [Float], precision: ImageGenerationPrecision) throws -> [Float] {
        guard noise.count == latentCount else { throw ImageGenerationError.invalidShape }
        return try noise.map(precision.rounded)
    }

    public static func modelTimestep(index: Int, precision: ImageGenerationPrecision) throws -> Float {
        guard sigmas.indices.contains(index) else { throw ImageGenerationError.invalidShape }
        return try precision.rounded(precision.rounded(sigmas[index] * 1_000) / 1_000)
    }

    public static func step(
        latents: [Float], prediction: [Float], index: Int, precision: ImageGenerationPrecision
    ) throws -> [Float] {
        guard latents.count == latentCount, prediction.count == latentCount, sigmas.indices.contains(index) else {
            throw ImageGenerationError.invalidShape
        }
        let next: Float = index + 1 < sigmas.count ? sigmas[index + 1] : 0
        let delta = try precision.rounded(next - sigmas[index])
        return try zip(latents, prediction).map { latent, predicted in
            guard latent.isFinite, predicted.isFinite else { throw ImageGenerationError.nonFinite }
            let increment = try precision.rounded(delta * precision.rounded(predicted))
            return try precision.rounded(latent + increment)
        }
    }
}

public struct ImageLatentScaling: Codable, Equatable, Sendable {
    public let mean: [Float]
    public let standardDeviation: [Float]

    public init(mean: [Float], standardDeviation: [Float]) throws {
        guard mean.count == 64, standardDeviation.count == 64,
            mean.allSatisfy(\.isFinite), standardDeviation.allSatisfy({ $0.isFinite && $0 > 0 })
        else { throw ImageGenerationError.invalidProfile("latent scaling") }
        self.mean = mean
        self.standardDeviation = standardDeviation
    }

    public func normalize(
        _ channelsFirst: [Float], precision: ImageGenerationPrecision = .float32
    ) throws -> [Float] {
        try validate(channelsFirst)
        var result = [Float](repeating: 0, count: ImageEuler.latentCount)
        for channel in 0..<64 {
            let offset = try precision.rounded(mean[channel])
            let scale = try precision.rounded(standardDeviation[channel])
            for token in 0..<1_024 {
                let value = try precision.rounded(channelsFirst[channel * 1_024 + token])
                let difference = try precision.rounded(value - offset)
                result[token * 64 + channel] = try precision.rounded(difference / scale)
            }
        }
        guard result.allSatisfy(\.isFinite) else { throw ImageGenerationError.nonFinite }
        return result
    }

    public func denormalize(
        _ tokensFirst: [Float], precision: ImageGenerationPrecision = .float32
    ) throws -> [Float] {
        try validate(tokensFirst)
        var result = [Float](repeating: 0, count: ImageEuler.latentCount)
        for channel in 0..<64 {
            let offset = try precision.rounded(mean[channel])
            let scale = try precision.rounded(standardDeviation[channel])
            for token in 0..<1_024 {
                let value = try precision.rounded(tokensFirst[token * 64 + channel])
                let product = try precision.rounded(value * scale)
                result[channel * 1_024 + token] = try precision.rounded(product + offset)
            }
        }
        guard result.allSatisfy(\.isFinite) else { throw ImageGenerationError.nonFinite }
        return result
    }

    private func validate(_ values: [Float]) throws {
        guard mean.count == 64, standardDeviation.count == 64, mean.allSatisfy(\.isFinite),
            standardDeviation.allSatisfy({ $0.isFinite && $0 > 0 }), values.count == ImageEuler.latentCount
        else { throw ImageGenerationError.invalidShape }
        guard values.allSatisfy(\.isFinite) else { throw ImageGenerationError.nonFinite }
    }
}

public enum ImageGenerationNoise {
    public static let version = "splitmix64-boxmuller-f64-packed-v1"

    public static func make(seed: UInt64) throws -> [Float] {
        guard seed <= Int64.max else { throw ImageGenerationError.invalidSeed }
        var state = seed
        func uniform() -> Double {
            state &+= 0x9E37_79B9_7F4A_7C15
            var value = state
            value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
            value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
            value ^= value >> 31
            return (Double(value >> 11) + 0.5) / 9_007_199_254_740_992
        }
        var values = [Float]()
        values.reserveCapacity(ImageEuler.latentCount)
        for _ in 0..<(ImageEuler.latentCount / 2) {
            let radius = sqrt(-2 * log(uniform()))
            let angle = 2 * Double.pi * uniform()
            values.append(Float(radius * cos(angle)))
            values.append(Float(radius * sin(angle)))
        }
        return values
    }
}
