import AVFoundation
import Foundation
import VideoSearchFeature

public enum VideoClipExporter {
    public static func export(
        _ plan: VideoClipPlan, of video: URL, to destination: URL, progress: @escaping @Sendable (Double) -> Void
    ) async throws -> Double {
        let accessed = video.startAccessingSecurityScopedResource()
        defer { if accessed { video.stopAccessingSecurityScopedResource() } }
        try? FileManager.default.removeItem(at: destination)
        do {
            let (session, seconds) = try await prepare(plan, of: video)
            let states = session.states(updateInterval: 0.2)
            try await withThrowingTaskGroup { group in
                group.addTask { for await case .exporting(let value) in states { progress(value.fractionCompleted) } }
                try await session.export(to: destination, as: .mp4)
            }
            return seconds
        } catch {
            try? FileManager.default.removeItem(at: destination)
            throw error
        }
    }

    private static func prepare(
        _ plan: VideoClipPlan, of video: URL
    ) async throws -> (session: AVAssetExportSession, seconds: Double) {
        let asset = AVURLAsset(url: video, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
        let tracks = try await asset.load(.tracks)
        guard let source = tracks.first(where: { $0.mediaType == .video }) else { throw VideoSearchError.invalidVideo }
        let audio = tracks.first { $0.mediaType == .audio }
        let (transform, available, timescale) = try await source.load(
            .preferredTransform, .timeRange, .naturalTimeScale)
        let heard = try await audio?.load(.segments).filter { !$0.isEmpty }.map(\.timeMapping.target) ?? []
        let composition = AVMutableComposition()
        guard
            let track = composition.addMutableTrack(
                withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid)
        else { throw VideoSearchError.invalidVideo }
        var audioTrack: AVMutableCompositionTrack?
        let scale = timescale > 0 ? timescale : 600
        var cursor = CMTime.zero
        for range in plan.ranges {
            let clip = CMTimeRange(
                start: CMTime(seconds: range.start, preferredTimescale: scale),
                end: CMTime(seconds: range.end, preferredTimescale: scale)
            ).intersection(available)
            guard clip.duration > .zero else { continue }
            try track.insertTimeRange(clip, of: source, at: cursor)
            for overlap in heard.map(clip.intersection) where overlap.duration > .zero {
                guard let audio else { break }
                audioTrack =
                    audioTrack
                    ?? composition.addMutableTrack(
                        withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)
                try audioTrack?.insertTimeRange(overlap, of: audio, at: cursor + overlap.start - clip.start)
            }
            cursor = cursor + clip.duration
        }
        guard cursor > .zero else { throw VideoSearchError.invalidVideo }
        track.preferredTransform = transform
        guard let session = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality)
        else { throw VideoSearchError.invalidVideo }
        session.shouldOptimizeForNetworkUse = true
        return (session, cursor.seconds)
    }
}
