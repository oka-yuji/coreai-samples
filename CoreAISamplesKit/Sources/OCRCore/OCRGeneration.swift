import Foundation

public struct OCRGeneration: Equatable, Sendable {
    public enum Finish: String, Equatable, Sendable {
        case stopToken
        case maxNewTokens
    }

    public static let repetitionPenalty: Float = 1.05
    public static let stopTokens: Set<Int32> = [151_645, 151_643]
    public static let defaultMaxNewTokens = 4_096

    public let maxNewTokens: Int
    public private(set) var tokens: [Int32] = []
    public private(set) var finish: Finish?
    private var seen: Set<Int32>

    public init(prompt: [Int32], maxNewTokens: Int = defaultMaxNewTokens) {
        self.maxNewTokens = maxNewTokens
        seen = Set(prompt)
    }

    public static func penalized(_ logits: [Float], seen: Set<Int32>) throws -> [Float] {
        guard logits.allSatisfy(\.isFinite) else { throw OCRError.nonFiniteLogits }
        guard seen.allSatisfy({ $0 >= 0 && Int($0) < logits.count }) else { throw OCRError.invalidShape }
        var result = logits
        for token in seen {
            let value = result[Int(token)]
            result[Int(token)] = value < 0 ? value * repetitionPenalty : value / repetitionPenalty
        }
        return result
    }

    public mutating func select(_ logits: [Float]) throws -> Int32 {
        guard finish == nil, tokens.count < maxNewTokens else { throw OCRError.generationFinished }
        let scores = try Self.penalized(logits, seen: seen)
        guard var best = scores.indices.first else { throw OCRError.invalidShape }
        for index in scores.indices where scores[index] > scores[best] { best = index }
        let token = Int32(best)
        tokens.append(token)
        seen.insert(token)
        if Self.stopTokens.contains(token) {
            finish = .stopToken
        } else if tokens.count == maxNewTokens {
            finish = .maxNewTokens
        }
        return token
    }
}
