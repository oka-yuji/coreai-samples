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
}
