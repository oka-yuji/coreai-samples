import Foundation

public enum EmbeddingError: Error, Equatable, Sendable {
    case invalidImage
    case invalidVector
    case invalidBundle(String)
    case tooManyTokens(Int)
    case invalidToken(Int)
    case busy
    case notLoaded
}

public struct EmbeddingVector: Equatable, Sendable {
    public let values: [Float]

    public init(_ values: [Float]) throws {
        guard values.count == 768, values.allSatisfy(\.isFinite) else { throw EmbeddingError.invalidVector }
        let norm = sqrt(values.reduce(0.0) { $0 + Double($1) * Double($1) })
        guard abs(norm - 1) <= 0.001 else { throw EmbeddingError.invalidVector }
        self.values = values
    }

    public func similarity(to other: Self) -> Double {
        let dot = zip(values, other.values).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
        let left = values.reduce(0.0) { $0 + Double($1) * Double($1) }
        let right = other.values.reduce(0.0) { $0 + Double($1) * Double($1) }
        return min(1, max(-1, dot / sqrt(left * right)))
    }
}

public enum EmbeddingTextRole: String, Codable, Sendable {
    case query
    case document

    public var prefix: String {
        switch self {
        case .query: "task: search result | query: "
        case .document: "title: none | text: "
        }
    }
}
