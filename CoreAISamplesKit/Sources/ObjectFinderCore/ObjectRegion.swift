import Foundation

public struct ObjectRegion: Codable, Equatable, Sendable {
    public let x: Double
    public let y: Double
    public let width: Double
    public let height: Double

    public init(x: Double, y: Double, width: Double, height: Double) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }

    public var isValid: Bool {
        [x, y, width, height].allSatisfy(\.isFinite)
            && x >= 0 && y >= 0 && width > 0 && height > 0
            && x + width <= 1 && y + height <= 1
    }

    public func overlap(with other: Self) -> Double {
        guard isValid, other.isValid else { return 0 }
        let intersection =
            max(0, min(x + width, other.x + other.width) - max(x, other.x))
            * max(0, min(y + height, other.y + other.height) - max(y, other.y))
        return intersection / (width * height + other.width * other.height - intersection)
    }
}

public struct ObjectCandidate: Codable, Equatable, Sendable, Identifiable {
    public let id: Int
    public let region: ObjectRegion
    public let similarity: Double

    public init(id: Int, region: ObjectRegion, similarity: Double) {
        self.id = id
        self.region = region
        self.similarity = similarity
    }
}

public enum ObjectCandidateSelection {
    public static func matching(
        _ candidates: [ObjectCandidate], minimumSimilarity: Double
    ) -> [ObjectCandidate] {
        guard minimumSimilarity.isFinite, (-1...1).contains(minimumSimilarity) else { return [] }
        return candidates.filter {
            $0.region.isValid && $0.similarity.isFinite
                && (-1.000_001...1.000_001).contains($0.similarity) && $0.similarity >= minimumSimilarity
        }
    }

    public static func select(
        _ candidates: [ObjectCandidate], minimumSimilarity: Double, limit: Int = 3
    ) -> [ObjectCandidate] {
        guard limit > 0 else { return [] }
        let ordered = matching(candidates, minimumSimilarity: minimumSimilarity).sorted {
            $0.similarity == $1.similarity ? $0.id < $1.id : $0.similarity > $1.similarity
        }
        var selected: [ObjectCandidate] = []
        for candidate in ordered {
            guard !selected.contains(where: { $0.region.overlap(with: candidate.region) > 0.5 }) else { continue }
            selected.append(candidate)
            if selected.count == limit { break }
        }
        return selected
    }
}
