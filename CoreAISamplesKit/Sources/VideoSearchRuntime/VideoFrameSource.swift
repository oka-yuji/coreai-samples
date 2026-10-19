import AVFoundation
import CoreGraphics
import Foundation

public enum VideoSearchError: Error, Equatable, Sendable {
    case invalidVideo
}

public enum VideoFrameSource {
    public static func duration(of url: URL) async throws -> Double {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let seconds = try await asset.load(.duration).seconds
        guard !tracks.isEmpty, seconds.isFinite, seconds > 0 else { throw VideoSearchError.invalidVideo }
        return seconds
    }

    public static func frames(of url: URL, at times: [Double], maximumSize: CGSize) -> VideoFrames {
        VideoFrames(url: url, times: times, maximumSize: maximumSize)
    }
}

public struct VideoFrames: AsyncSequence {
    public typealias Element = (time: Double, image: CGImage)

    let url: URL
    let times: [Double]
    let maximumSize: CGSize

    public func makeAsyncIterator() -> Iterator {
        let access = GeneratorAccess(url: url, maximumSize: maximumSize)
        let requests = times.map { CMTime(seconds: $0, preferredTimescale: 600) }
        return Iterator(access: access, pending: requests[...])
    }

    public struct Iterator: AsyncIteratorProtocol {
        static let batch = 4
        let access: GeneratorAccess
        var pending: ArraySlice<CMTime>
        var images: AVAssetImageGenerator.Images?

        public mutating func next() async throws -> (time: Double, image: CGImage)? {
            while true {
                try checkCancellation()
                if images == nil {
                    guard !pending.isEmpty else { return nil }
                    images = access.generator.images(for: Array(pending.prefix(Self.batch)))
                    pending = pending.dropFirst(Self.batch)
                }
                guard let element = await images?.next(isolation: #isolation) else {
                    images = nil
                    continue
                }
                try checkCancellation()
                guard case .success(_, let image, let actualTime) = element else { continue }
                return (actualTime.seconds, image)
            }
        }

        private func checkCancellation() throws {
            guard Task.isCancelled else { return }
            access.generator.cancelAllCGImageGeneration()
            throw CancellationError()
        }
    }
}

final class GeneratorAccess {
    let generator: AVAssetImageGenerator
    private let url: URL
    private let accessed: Bool

    init(url: URL, maximumSize: CGSize) {
        self.url = url
        accessed = url.startAccessingSecurityScopedResource()
        generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = maximumSize
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.5, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.5, preferredTimescale: 600)
    }

    deinit {
        generator.cancelAllCGImageGeneration()
        if accessed { url.stopAccessingSecurityScopedResource() }
    }
}
