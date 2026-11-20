import AVFoundation
import Foundation

@MainActor
public final class SpokenRecordingPlayer {
    private let muted: Bool
    private var player: AVPlayer?
    private var pending: AsyncStream<Bool>.Continuation?

    public init(muted: Bool = false) { self.muted = muted }

    public func play(_ url: URL, from seconds: Double, onStarted: @MainActor (Double) -> Void) async throws {
        stop()
        try Task.checkCancellation()
        guard seconds.isFinite, seconds >= 0 else { throw SpokenPlayerError.cannotPlay }
        let item = AVPlayerItem(url: url)
        let candidate = AVPlayer(playerItem: item)
        candidate.isMuted = muted
        let (events, continuation) = AsyncStream<Bool>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let center = NotificationCenter.default
        let end = center.addObserver(forName: AVPlayerItem.didPlayToEndTimeNotification, object: item, queue: nil) {
            _ in
            continuation.yield(true)
            continuation.finish()
        }
        let failure = center.addObserver(
            forName: AVPlayerItem.failedToPlayToEndTimeNotification, object: item, queue: nil
        ) { _ in
            continuation.yield(false)
            continuation.finish()
        }
        player = candidate
        pending = continuation
        defer {
            center.removeObserver(end)
            center.removeObserver(failure)
            if player === candidate { stop() }
        }
        try await withTaskCancellationHandler {
            guard try await item.asset.load(.isPlayable) else { throw SpokenPlayerError.cannotPlay }
            try Task.checkCancellation()
            guard player === candidate else { throw CancellationError() }
            let time = CMTime(seconds: seconds, preferredTimescale: 48_000)
            let finished = await candidate.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero)
            try Task.checkCancellation()
            guard player === candidate else { throw CancellationError() }
            guard finished, item.status != .failed else { throw SpokenPlayerError.cannotPlay }
            candidate.play()
            onStarted(candidate.currentTime().seconds)
            for await success in events {
                if !success { throw SpokenPlayerError.cannotPlay }
                return
            }
            throw CancellationError()
        } onCancel: {
            continuation.finish()
        }
    }

    public func stop() {
        player?.pause()
        player?.currentItem?.cancelPendingSeeks()
        player?.replaceCurrentItem(with: nil)
        player = nil
        pending?.finish()
        pending = nil
    }
}
