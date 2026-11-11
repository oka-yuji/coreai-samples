import Foundation

public struct ChronosGeoPoint: Codable, Equatable, Sendable {
    public let latitude: Double
    public let longitude: Double

    public init(latitude: Double, longitude: Double) {
        self.latitude = latitude
        self.longitude = longitude
    }

    public var isValid: Bool {
        latitude.isFinite && longitude.isFinite && (-90...90).contains(latitude) && (-180...180).contains(longitude)
    }
}

public struct ChronosZonePolygon: Codable, Equatable, Sendable {
    public let outer: [ChronosGeoPoint]
    public let holes: [[ChronosGeoPoint]]

    public init(outer: [ChronosGeoPoint], holes: [[ChronosGeoPoint]] = []) {
        self.outer = outer
        self.holes = holes
    }

    public var isValid: Bool {
        ([outer] + holes).allSatisfy { $0.count >= 4 && $0.first == $0.last && $0.allSatisfy(\.isValid) }
    }

    public func contains(_ point: ChronosGeoPoint) -> Bool {
        inside(point, ring: outer) && !holes.contains { inside(point, ring: $0) }
    }

    private func inside(_ point: ChronosGeoPoint, ring: [ChronosGeoPoint]) -> Bool {
        guard ring.count >= 4 else { return false }
        var result = false
        for index in 1..<ring.count {
            let a = ring[index - 1]
            let b = ring[index]
            if (a.latitude > point.latitude) != (b.latitude > point.latitude) {
                let crossing =
                    (b.longitude - a.longitude) * (point.latitude - a.latitude)
                    / (b.latitude - a.latitude) + a.longitude
                if point.longitude < crossing { result.toggle() }
            }
        }
        return result
    }
}
