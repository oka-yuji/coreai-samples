public enum ObjectTrackingPolicy {
    public static let maximumAge = 2.0
    public static let minimumConfidence = 0.5

    public static func accepts(capturedAt: Double, now: Double, generation: Int, currentGeneration: Int) -> Bool {
        capturedAt.isFinite && now.isFinite && now >= capturedAt && now - capturedAt <= maximumAge
            && generation == currentGeneration
    }

    public static func accepts(region: ObjectRegion, confidence: Double) -> Bool {
        region.isValid && confidence.isFinite && (minimumConfidence...1).contains(confidence)
    }

    public static func metadataRegion(_ region: ObjectRegion, rotation: Double) -> ObjectRegion? {
        guard region.isValid else { return nil }
        switch rotation {
        case 0: return region
        case 90:
            return ObjectRegion(x: region.y, y: 1 - region.x - region.width, width: region.height, height: region.width)
        case 180:
            return ObjectRegion(
                x: 1 - region.x - region.width, y: 1 - region.y - region.height, width: region.width,
                height: region.height)
        case 270:
            return ObjectRegion(
                x: 1 - region.y - region.height, y: region.x, width: region.height, height: region.width)
        default: return nil
        }
    }
}
