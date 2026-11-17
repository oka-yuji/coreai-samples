import Foundation
import SpeechCore
import Testing

@testable import SpeechRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct SpeechEngineTests {
    @Test("A service without an engine runs Irodori under the Irodori key and VoxCPM2 has its own runtime and key")
    func engineSelection() throws {
        let name = "SpeechEngineTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let standard = LiveSpeechService(defaults: defaults)
        #expect(standard.engine is IrodoriRuntime)
        #expect(standard.modelPathKey == "irodoriModelPath")
        #expect(LiveSpeechService.modelPathKey == "irodoriModelPath")
        let irodori = LiveSpeechService(engine: .irodori, defaults: defaults)
        #expect(irodori.engine is IrodoriRuntime)
        #expect(irodori.modelPathKey == "irodoriModelPath")
        let voxcpm = LiveSpeechService(engine: .voxcpm2, defaults: defaults)
        #expect(voxcpm.engine is VoxCPMRuntime)
        #expect(voxcpm.modelPathKey == "voxcpm2ModelPath")
    }

    @Test("Opening a VoxCPM2 model saves its folder under the VoxCPM2 key and leaves the Irodori key alone")
    func separatePathKeys() async throws {
        let name = "SpeechEngineTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        defaults.set("/irodori", forKey: SpeechEngine.irodori.modelPathKey)
        let service = LiveSpeechService(
            engine: FakeSpeechEngine(), playback: AudioPlayback(), defaults: defaults,
            modelPathKey: SpeechEngine.voxcpm2.modelPathKey)
        _ = try await service.open(path: "/voxcpm2")
        #expect(defaults.string(forKey: "voxcpm2ModelPath") == "/voxcpm2")
        #expect(defaults.string(forKey: "irodoriModelPath") == "/irodori")
    }
}
