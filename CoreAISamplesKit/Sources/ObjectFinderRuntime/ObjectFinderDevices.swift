import CoreGraphics
import Foundation
import ObjectFinderCore

struct CapturedObjectFrame: Sendable {
    let image: CGImage
    let jpeg: Data
    let rotation: Double
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
}
