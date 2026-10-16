import CoreGraphics
import Foundation
import ObjectFinderCore

struct CapturedObjectFrame: Sendable {
    let image: CGImage
    let rotation: Double
    var generation: Int = 0
    var capturedAt: Double = 0
}

@MainActor
struct ObjectFinderDevices {
    var authorize: () async throws -> Void
    var prepare: (URL, String) async throws -> Void
    var startCamera: (Double) async throws -> AsyncThrowingStream<CapturedObjectFrame, any Error>
    var stopCamera: () async -> Void
    var checkCamera: () async throws -> Void
    var candidates: (CGImage) async throws -> [ObjectCandidate]
    var closeFinder: () async throws -> Void
    var beginOrientation: () -> Void
    var endOrientation: () -> Void
    var notify: () -> Void
    var trackCandidates: (CapturedObjectFrame, [ObjectCandidate]) async -> [ObjectCandidate] = { _, candidates in
        candidates
    }
}
