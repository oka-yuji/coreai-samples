#if os(macOS)
    import AVFAudio
    import Foundation
    import SpeechCore

    @MainActor
    protocol PlaybackDevice: AnyObject {
        var completion: ((Bool) -> Void)? { get set }
        func play() -> Bool
        func stop()
    }

    @MainActor
    final class PlayerDevice: NSObject, PlaybackDevice, AVAudioPlayerDelegate {
        private let player: AVAudioPlayer
        var completion: ((Bool) -> Void)?

        init(data: Data) throws {
            player = try AVAudioPlayer(data: data)
            super.init()
            player.delegate = self
        }
        func play() -> Bool { player.prepareToPlay() && player.play() }
        func stop() { player.stop() }
        func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) { completion?(flag) }
        func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: (any Error)?) { completion?(false) }
    }

    @MainActor
    final class AudioPlayback {
        private let makeDevice: (Data) throws -> any PlaybackDevice
        private var device: (any PlaybackDevice)?
        private var continuation: CheckedContinuation<Void, any Error>?
        private var generation = 0

        init(makeDevice: @escaping (Data) throws -> any PlaybackDevice = { try PlayerDevice(data: $0) }) {
            self.makeDevice = makeDevice
        }

        func play(_ data: Data) async throws {
            guard continuation == nil else { throw SpeechError.busy }
            try Task.checkCancellation()
            generation += 1
            let ticket = generation
            let created = try makeDevice(data)
            try await withTaskCancellationHandler {
                try Task.checkCancellation()
                try await withCheckedThrowingContinuation { pending in
                    continuation = pending
                    device = created
                    created.completion = { [weak self] success in
                        self?.finish(ticket, result: success ? .success(()) : .failure(SpeechError.playbackFailed))
                    }
                    if !created.play() { finish(ticket, result: .failure(SpeechError.playbackFailed)) }
                }
            } onCancel: {
                DispatchQueue.main.async { [weak self] in
                    self?.finish(ticket, result: .failure(CancellationError()))
                }
            }
        }

        func stop() { finish(generation, result: .failure(CancellationError())) }

        private func finish(_ ticket: Int, result: Result<Void, any Error>) {
            guard ticket == generation, let pending = continuation else { return }
            continuation = nil
            let previous = device
            device = nil
            previous?.completion = nil
            previous?.stop()
            pending.resume(with: result)
        }
    }
#endif
