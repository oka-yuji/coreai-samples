#if os(iOS)
    import CoreImage
    import Foundation
    import ObjectFinderCore
    import Vision

    final class ObjectFinderTracker {
        private let handler = VNSequenceRequestHandler()
        private var requests: [(ObjectCandidate, VNTrackObjectRequest)] = []
        let capturedAt: Double

        init(candidates: [ObjectCandidate], frame: CapturedObjectFrame) throws {
            capturedAt = frame.capturedAt
            for candidate in candidates.prefix(3) where candidate.region.isValid && candidate.similarity.isFinite {
                let box = candidate.region
                let observation = VNDetectedObjectObservation(
                    boundingBox: CGRect(x: box.x, y: 1 - box.y - box.height, width: box.width, height: box.height))
                let request = VNTrackObjectRequest(detectedObjectObservation: observation)
                request.revision = VNTrackObjectRequestRevision1
                request.trackingLevel = .fast
                requests.append((candidate, request))
            }
            if let request = requests.first?.1 {
                var error: NSError?
                let count = request.supportedNumber(ofTrackersAndReturnError: &error)
                if let error { throw error }
                requests = Array(requests.prefix(Int(clamping: count)))
            }
            if !requests.isEmpty { try handler.perform(requests.map(\.1), on: frame.image) }
            _ = collect()
        }

        func track(_ image: CIImage, now: Double) throws -> [ObjectCandidate] {
            guard ObjectTrackingPolicy.accepts(capturedAt: capturedAt, now: now, generation: 0, currentGeneration: 0)
            else {
                requests = []
                return []
            }
            guard !requests.isEmpty else { return [] }
            try handler.perform(requests.map(\.1), on: image)
            return collect()
        }

        private func collect() -> [ObjectCandidate] {
            var next: [(ObjectCandidate, VNTrackObjectRequest)] = []
            var candidates: [ObjectCandidate] = []
            for (candidate, request) in requests {
                guard let result = request.results?.first as? VNDetectedObjectObservation else { continue }
                let box = result.boundingBox
                let region = ObjectRegion(x: box.minX, y: 1 - box.maxY, width: box.width, height: box.height)
                guard ObjectTrackingPolicy.accepts(region: region, confidence: Double(result.confidence)) else {
                    continue
                }
                request.inputObservation = result
                next.append((candidate, request))
                candidates.append(ObjectCandidate(id: candidate.id, region: region, similarity: candidate.similarity))
            }
            requests = next
            return candidates
        }
    }
#endif
