import AVFAudio
import Foundation
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
