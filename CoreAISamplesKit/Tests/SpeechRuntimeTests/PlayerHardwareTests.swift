import Foundation
import SpeechCore
import Testing

@testable import SpeechRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct PlayerHardwareTests {
    @Test("A player that has been stopped refuses to start")
    func stopBeforePlay() async throws {
        let wav = try floatWAV(samples: [Float](repeating: 0, count: 4_800))
        let delegate = try PlayerDevice(data: wav)
        let hardware = try PlayerHardware(data: wav)
        await hardware.stop()
        let started = await hardware.play(delegate: delegate)
        #expect(!started)
    }

    @Test("A short WAV starts on the player and then stops")
    func playThenStop() async throws {
        let wav = try floatWAV(samples: [Float](repeating: 0, count: 4_800))
        let delegate = try PlayerDevice(data: wav)
        let hardware = try PlayerHardware(data: wav)
        let started = await hardware.play(delegate: delegate)
        await hardware.stop()
        #expect(started)
    }

    @Test("Playback on the real player finishes through its delegate and can play again")
    func naturalFinishThenReplay() async throws {
        let wav = try floatWAV(samples: [Float](repeating: 0, count: 4_800))
        let playback = AudioPlayback(session: PassivePlaybackSession()) { try PlayerDevice(data: $0) }
        try await playback.play(wav)
        try await playback.play(wav)
    }

    @Test("Stopping playback on the real player right after it starts ends it once with cancellation")
    func stopRealPlayback() async throws {
        let wav = try floatWAV(samples: [Float](repeating: 0, count: 48_000))
        let created = TestSignal()
        let playback = AudioPlayback(session: PassivePlaybackSession()) { data in
            created.fire()
            return try PlayerDevice(data: data)
        }
        let task = Task { try await playback.play(wav) }
        defer {
            playback.stop()
            task.cancel()
        }
        try await created.wait()
        playback.stop()
        await #expect(throws: CancellationError.self) { try await task.value }
    }
}
