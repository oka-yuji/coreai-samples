import Foundation
import SpokenSearchFeature

@MainActor
final class SpokenSearchRunner {
    private let isPermitted: @MainActor () -> Bool
    private let speech: any SpokenTranscribing
    private let player = SpokenRecordingPlayer()
    private var recording: SpokenRecording?
    private var generation = 0
    private var closed = false
    private var isClosing = false
    private var operations = 0
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private var closeWaiters: [CheckedContinuation<Void, Never>] = []

    init(
        isPermitted: @escaping @MainActor () -> Bool,
        speech: any SpokenTranscribing = SpokenFileTranscriber()
    ) {
        self.isPermitted = isPermitted
        self.speech = speech
    }

    func run(_ effect: SpokenSearchEffect, send: (SpokenSearchAction) -> Void) async {
        guard !closed else { return }
        operations += 1
        defer { finished() }
        switch effect {
        case .open(let id, let url): await open(id, url: url, send: send)
        case .clear(let id):
            guard id >= generation else { return }
            generation = id
            await clear()
        case .stopPlayback(let id):
            if id == generation { await player.stopAndWait() }
        case .find(let id, let query):
            guard let record = current(id), let transcript = record.transcript else {
                send(.failed(id, .unavailable))
                return
            }
            let hits = await Self.matches(in: transcript, query: query)
            if current(id) === record { send(.found(id, hits)) }
        case .play(let id, let hit):
            guard let record = current(id) else {
                send(.failed(id, .unavailable))
                return
            }
            do {
                try await player.play(record.url, from: hit.start) { _ in }
                if current(id) === record { send(.playbackEnded(id, hit.id)) }
            } catch {
                if current(id) === record { send(.failed(id, .playbackFailed)) }
            }
        }
    }

    func close() async {
        if closed {
            if isClosing { await withCheckedContinuation { closeWaiters.append($0) } }
            return
        }
        closed = true
        isClosing = true
        await clear()
        if operations > 0 { await withCheckedContinuation { waiters.append($0) } }
        isClosing = false
        let completed = closeWaiters
        closeWaiters.removeAll()
        for waiter in completed { waiter.resume() }
    }

    private func open(_ id: Int, url: URL, send: (SpokenSearchAction) -> Void) async {
        guard id >= generation else { return }
        generation = id
        guard isPermitted() else {
            await clear()
            send(.failed(id, .unavailable))
            return
        }
        player.stop()
        let record = SpokenRecording(url, generation: id)
        recording = record
        await speech.cancel()
        guard current(id) === record else { return }
        do {
            let transcript = try await speech.transcribe(url)
            guard current(id) === record else { return }
            guard transcript.runs.contains(where: { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })
            else {
                recording = nil
                send(.failed(id, .noSpeech))
                return
            }
            record.transcript = transcript
            send(.recognized(id))
        } catch {
            guard current(id) === record else { return }
            recording = nil
            send(.failed(id, Self.notice(for: error)))
        }
    }

    private func current(_ id: Int) -> SpokenRecording? {
        guard !closed, !Task.isCancelled, isPermitted(), recording?.generation == id else { return nil }
        return recording
    }

    private func clear() async {
        recording = nil
        let playbackStop = player.beginStopping()
        await speech.cancel()
        await playbackStop.wait()
    }

    private func finished() {
        operations -= 1
        if operations == 0 {
            let pending = waiters
            waiters.removeAll()
            for waiter in pending { waiter.resume() }
        }
    }

    @concurrent private static func matches(in transcript: SpokenTranscript, query: String) async -> [SpokenHit] {
        transcript.matches(query)
    }

    private static func notice(for error: any Error) -> SpokenNotice {
        switch error as? SpokenRecognitionError {
        case .unavailable: .unavailable
        case .languageUnavailable: .languageUnavailable
        case .unreadableFile: .unreadableFile
        default: .recognitionFailed
        }
    }
}
