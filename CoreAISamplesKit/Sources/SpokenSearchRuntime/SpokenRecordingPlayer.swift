import AVFoundation
import Foundation
import SpeechRuntime

@MainActor
public final class SpokenRecordingPlayer {
    private let muted: Bool
    private let session: any PlaybackSession
    private var generation = 0
    private var player: AVPlayer?
    private var pending: AsyncStream<Bool>.Continuation?

    public convenience init(muted: Bool = false) { self.init(muted: muted, session: PlaybackSessions.make()) }

    init(muted: Bool, session: any PlaybackSession) {
        self.muted = muted
        self.session = session
    }

    public func play(_ url: URL, from seconds: Double, onStarted: @MainActor (Double) -> Void) async throws {
        let prior = beginStopping()
        let ticket = generation
        await prior.wait()
        try Task.checkCancellation()
        guard generation == ticket else { throw CancellationError() }
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
        }
        do {
            try await withTaskCancellationHandler {
                guard try await item.asset.load(.isPlayable) else { throw SpokenPlayerError.cannotPlay }
                try check(ticket)
                let time = CMTime(seconds: seconds, preferredTimescale: 48_000)
                let finished = await candidate.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero)
                try check(ticket)
                guard finished, item.status != .failed else { throw SpokenPlayerError.cannotPlay }
                try await activate(ticket)
                try check(ticket)
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
        } catch {
            if generation == ticket { await stopAndWait() }
            throw error
        }
        if generation == ticket { await stopAndWait() }
    }

    public func stop() {
        _ = beginStopping()
    }

    func beginStopping() -> SpokenPlaybackStop {
        let completion = SpokenPlaybackStop()
        generation += 1
        player?.pause()
        player?.currentItem?.cancelPendingSeeks()
        player?.replaceCurrentItem(with: nil)
        player = nil
        pending?.finish()
        pending = nil
        session.stop { completion.finish() }
        return completion
    }

    func stopAndWait() async {
        await beginStopping().wait()
    }

    private func activate(_ ticket: Int) async throws {
        try await withCheckedThrowingContinuation { ready in
            session.start(
                interrupted: { [weak self] in
                    guard let self, generation == ticket else { return }
                    stop()
                }, ready: { ready.resume(with: $0) })
        }
    }

    private func check(_ ticket: Int) throws {
        try Task.checkCancellation()
        if generation != ticket { throw CancellationError() }
    }
}
