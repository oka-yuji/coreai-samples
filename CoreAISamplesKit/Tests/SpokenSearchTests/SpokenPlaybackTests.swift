import AVFAudio
import Foundation
import SpokenSearchFeature
import Testing

@testable import SpokenSearchRuntime

@MainActor
struct SpokenPlaybackTests {
    @Test("File playback activates the speech playback session before starting and waits for its deactivation")
    func audioSessionOrder() async throws {
        let url = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString + ".caf")
        defer { try? FileManager.default.removeItem(at: url) }
        try makeAudio(at: url)
        let session = SpokenPlaybackProbe()
        let player = SpokenRecordingPlayer(muted: true, session: session)
        var startedActive = false
        var finished = false
        let playing = Task {
            try await player.play(url, from: 0.02) { _ in startedActive = session.isActive }
            finished = true
        }
        await session.waitForDeactivation()
        #expect(!finished && startedActive)
        session.acknowledge()
        try await playing.value
        #expect(finished && !session.isActive && session.starts == 1)
    }

    @Test("An external audio interruption ends the selected playback without reporting a broken recording")
    func interruption() async throws {
        let url = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString + ".caf")
        defer { try? FileManager.default.removeItem(at: url) }
        try makeAudio(at: url)
        let speech = SuspendedTranscription()
        let player = SpokenRecordingPlayer(muted: true, session: SpokenPlaybackProbe(interruptsOnStart: true))
        let runner = SpokenSearchRunner(isPermitted: { true }, speech: speech, player: player)
        let opening = Task { await runner.run(.open(1, url)) { _ in } }
        await speech.waitForStart()
        speech.finish()
        await opening.value
        var ended = false
        var failed = false
        let hit = SpokenHit(id: 8, text: "day", start: 0.02, end: 0.04)
        await runner.run(.play(1, hit)) { action in
            if case .playbackEnded(1, 8) = action { ended = true }
            if case .failed = action { failed = true }
        }
        #expect(ended && !failed)
        await runner.close()
    }

    private func makeAudio(at url: URL) throws {
        let format = try #require(AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1600))
        buffer.frameLength = 1600
        let samples = try #require(buffer.floatChannelData?[0])
        samples.update(repeating: 0, count: 1600)
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
    }
}
