import Foundation
import ObjectFinderCore
import Testing

struct ObjectRegionTests {
    private let left = ObjectRegion(x: 0.1, y: 0.2, width: 0.3, height: 0.4)

    @Test("Looking down at a table preserves the last valid camera orientation")
    func flatOrientation() {
        var orientation = ObjectCameraOrientation()
        orientation.update(.flat)
        #expect(orientation.angle == 90)
        orientation.update(.landscapeLeft)
        orientation.update(.flat)
        #expect(orientation.angle == 0)
        orientation.update(.landscapeRight)
        orientation.update(.flat)
        #expect(orientation.angle == 180)
        orientation.update(.portraitUpsideDown)
        #expect(orientation.angle == 270)
        orientation.update(.portrait)
        #expect(orientation.angle == 90)
    }

    @Test("Invalid, non-finite and out-of-frame regions cannot become visible matches")
    func invalidRegions() {
        let invalid = [
            ObjectRegion(x: .nan, y: 0, width: 1, height: 1),
            ObjectRegion(x: 0, y: .infinity, width: 1, height: 1),
            ObjectRegion(x: -0.1, y: 0, width: 0.2, height: 0.2),
            ObjectRegion(x: 0.8, y: 0, width: 0.3, height: 0.2),
            ObjectRegion(x: 0, y: 0, width: 0, height: 1),
        ]
        for region in invalid {
            #expect(!region.isValid)
            #expect(
                ObjectCandidateSelection.select([.init(id: 0, region: region, similarity: 0.9)], minimumSimilarity: 0.5)
                    .isEmpty)
        }
        for score in [Double.nan, .infinity, -.infinity, 1.1, -1.1] {
            #expect(
                ObjectCandidateSelection.select([.init(id: 0, region: left, similarity: score)], minimumSimilarity: 0.5)
                    .isEmpty)
        }
    }

    @Test("Region filtering preserves coordinates, suppresses overlaps and permits an empty result below the threshold")
    func filtering() {
        let first = ObjectCandidate(id: 0, region: left, similarity: 0.8)
        let duplicate = ObjectCandidate(id: 1, region: left, similarity: 0.7)
        let right = ObjectCandidate(id: 2, region: .init(x: 0.6, y: 0.2, width: 0.3, height: 0.4), similarity: 0.65)
        #expect(ObjectCandidateSelection.select([duplicate, right, first], minimumSimilarity: 0.65) == [first, right])
        #expect(ObjectCandidateSelection.select([first], minimumSimilarity: 0.81).isEmpty)
        #expect(ObjectCandidateSelection.select([first, right], minimumSimilarity: 0.65, limit: 1) == [first])
        #expect(ObjectCandidateSelection.select([first], minimumSimilarity: .nan).isEmpty)
        #expect(ObjectCandidateSelection.select([first], minimumSimilarity: 0, limit: 0).isEmpty)
        #expect(abs(left.overlap(with: left) - 1) < 0.000_000_001)
        #expect(left.overlap(with: right.region) == 0)
    }

    @Test("Tracking eligibility preserves input order and overlapping matches until their current positions are known")
    func trackingEligibility() {
        let first = ObjectCandidate(id: 7, region: left, similarity: 0.8)
        let second = ObjectCandidate(id: 3, region: left, similarity: 0.9)
        let low = ObjectCandidate(id: 1, region: left, similarity: 0.7)
        let invalid = ObjectCandidate(id: 2, region: left, similarity: .nan)
        #expect(
            ObjectCandidateSelection.matching([low, first, invalid, second], minimumSimilarity: 0.8) == [first, second])
        #expect(ObjectCandidateSelection.matching([first], minimumSimilarity: .nan).isEmpty)
        #expect(ObjectCandidateSelection.matching([first], minimumSimilarity: 2).isEmpty)
    }
}
