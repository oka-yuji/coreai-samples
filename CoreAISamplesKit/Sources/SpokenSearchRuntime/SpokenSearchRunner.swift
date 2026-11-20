import EmbeddingCore
import Foundation
import SpokenSearchFeature

@MainActor
final class SpokenSearchRunner {
    private let isPermitted: @MainActor () -> Bool
    private let speech: any SpokenTranscribing
    private let player: SpokenRecordingPlayer
    private let makeTopicIndex: @MainActor (URL) -> SpokenTopicIndex
    private var recording: SpokenRecording?
    private var topicSource: SpokenTopicSource?
    private var topicIndex: SpokenTopicIndex?
    private var searchingTopics = false
    private var topicWaiters: [CheckedContinuation<Void, Never>] = []
    private var generation = 0
    private var closed = false
    private var isClosing = false
    private var operations = 0
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private var closeWaiters: [CheckedContinuation<Void, Never>] = []

    init(
        isPermitted: @escaping @MainActor () -> Bool,
        speech: any SpokenTranscribing = SpokenFileTranscriber(),
        player: SpokenRecordingPlayer = SpokenRecordingPlayer(),
        makeTopicIndex: @escaping @MainActor (URL) -> SpokenTopicIndex = { SpokenTopicIndex(folder: $0) }
    ) {
        self.isPermitted = isPermitted
        self.speech = speech
        self.player = player
        self.makeTopicIndex = makeTopicIndex
    }

    func setTopicSource(_ source: SpokenTopicSource?) {
        guard !closed else { return }
        if source?.identity != topicSource?.identity || source?.directory != topicSource?.directory {
            topicIndex = source.map { makeTopicIndex($0.directory) }
        }
        topicSource = source
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
        case .findTopic(let id, let query, let identity):
            await findTopic(id, query: query, identity: identity, send: send)
        case .waitForTopic(let id):
            await waitForTopics()
            if !Task.isCancelled { send(.topicStopped(id)) }
        case .play(let id, let hit, let identity):
            if let identity, topicSource?.identity != identity || topicSource?.isCurrent() != true {
                send(.failed(id, .topicUnavailable))
                return
            }
            guard let record = current(id) else {
                send(.failed(id, .unavailable))
                return
            }
            do {
                try await player.play(record.url, from: hit.start) { _ in }
            } catch {
                if !(error is CancellationError) {
                    if current(id) === record { send(.failed(id, .playbackFailed)) }
                    return
                }
            }
            if current(id) === record { send(.playbackEnded(id, hit.id)) }
        }
    }

    func stopPlayback() { player.stop() }

    func close() async {
        if closed {
            if isClosing { await withCheckedContinuation { closeWaiters.append($0) } }
            return
        }
        closed = true
        topicSource = nil
        topicIndex = nil
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
        await topicIndex?.invalidate(before: id)
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

    private func findTopic(_ id: Int, query: String, identity: String, send: (SpokenSearchAction) -> Void) async {
        await waitForTopics()
        guard !Task.isCancelled else { return }
        guard let record = current(id), let transcript = record.transcript, let source = topicSource,
            source.identity == identity, source.isCurrent(), let index = topicIndex
        else {
            send(.failed(id, .topicUnavailable))
            return
        }
        searchingTopics = true
        defer {
            searchingTopics = false
            let pending = topicWaiters
            topicWaiters.removeAll()
            for waiter in pending { waiter.resume() }
        }
        func isCurrent() -> Bool {
            current(id) === record && topicIndex === index && topicSource?.identity == identity && source.isCurrent()
        }
        let release: SpokenTopicSource.Release
        do {
            release = try await source.acquire()
        } catch {
            if isCurrent() { send(.failed(id, .topicUnavailable)) }
            return
        }
        guard isCurrent() else {
            await release()
            return
        }
        let result: SpokenSearchAction
        do {
            let hits = try await index.search(query, transcript: transcript, generation: id)
            result = .topicFound(id, identity, hits)
        } catch EmbeddingError.tooManyTokens {
            result = .failed(id, .shortenQuery)
        } catch {
            result = .failed(id, .topicFailed)
        }
        await release()
        if isCurrent() { send(result) }
    }

    private func waitForTopics() async {
        while searchingTopics { await withCheckedContinuation { topicWaiters.append($0) } }
    }

    private func current(_ id: Int) -> SpokenRecording? {
        guard !closed, !Task.isCancelled, isPermitted(), recording?.generation == id else { return nil }
        return recording
    }

    private func clear() async {
        recording = nil
        let playbackStop = player.beginStopping()
        let index = topicIndex
        let lowerBound = generation
        await speech.cancel()
        await index?.invalidate(before: lowerBound)
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
