import ObjectFinderCore
import Testing

struct ObjectTrackingTests {
    @Test("Invalid, low-confidence, expired and old-generation tracking updates cannot leave visible boxes")
    func trackingGuards() {
        let region = ObjectRegion(x: 0.1, y: 0.2, width: 0.3, height: 0.4)
        #expect(ObjectTrackingPolicy.accepts(region: region, confidence: 0.9))
        #expect(!ObjectTrackingPolicy.accepts(region: region, confidence: 0.4))
        #expect(!ObjectTrackingPolicy.accepts(region: region, confidence: .nan))
        #expect(!ObjectTrackingPolicy.accepts(region: .init(x: -0.1, y: 0, width: 1, height: 1), confidence: 0.9))
        #expect(ObjectTrackingPolicy.accepts(capturedAt: 10, now: 11, generation: 2, currentGeneration: 2))
        #expect(!ObjectTrackingPolicy.accepts(capturedAt: 10, now: 13, generation: 2, currentGeneration: 2))
        #expect(!ObjectTrackingPolicy.accepts(capturedAt: 10, now: 11, generation: 1, currentGeneration: 2))
        #expect(!ObjectTrackingPolicy.accepts(capturedAt: 10, now: 9, generation: 2, currentGeneration: 2))
        #expect(!ObjectTrackingPolicy.accepts(capturedAt: .nan, now: 11, generation: 2, currentGeneration: 2))
    }

    @Test("Preview metadata boxes preserve the object across all camera orientations")
    func previewCoordinates() throws {
        let region = ObjectRegion(x: 0.1, y: 0.2, width: 0.3, height: 0.4)
        let expected: [(Double, [Double])] = [
            (0, [0.1, 0.2, 0.3, 0.4]), (90, [0.2, 0.6, 0.4, 0.3]), (180, [0.6, 0.4, 0.3, 0.4]),
            (270, [0.4, 0.1, 0.4, 0.3]),
        ]
        for (rotation, values) in expected {
            let box = try #require(ObjectTrackingPolicy.metadataRegion(region, rotation: rotation))
            for (actual, expected) in zip([box.x, box.y, box.width, box.height], values) {
                #expect(abs(actual - expected) < 0.000_001)
            }
        }
        #expect(ObjectTrackingPolicy.metadataRegion(region, rotation: 45) == nil)
    }
}
