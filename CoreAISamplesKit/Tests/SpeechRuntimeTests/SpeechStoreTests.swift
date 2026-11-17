import Foundation
import SpeechFeature
import SpeechRuntime
import Testing

@MainActor
struct SpeechStoreTests {
    @Test(
        "VUI-05 The VoxCPM2 store starts from the VoxCPM2 folder and its examples, and the Irodori store keeps its own")
    func separateStores() throws {
        let name = "SpeechStoreTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        defaults.set("/models/irodori", forKey: "irodoriModelPath")
        defaults.set("/models/voxcpm2", forKey: "voxcpm2ModelPath")
        let described = [SpeechExample(id: "calm", title: "Calm", text: "(A calm voice)Hello.")]
        let plain = [SpeechExample(id: "plain", title: "Plain", text: "Hello.")]
        let voxcpm2 = LiveSpeechService.makeStore(engine: .voxcpm2, examples: described, defaults: defaults)
        let irodori = LiveSpeechService.makeStore(engine: .irodori, examples: plain, defaults: defaults)
        #expect(voxcpm2.state.lastPath == "/models/voxcpm2" && irodori.state.lastPath == "/models/irodori")
        #expect(voxcpm2.state.examples == described && voxcpm2.state.text == "(A calm voice)Hello.")
        #expect(irodori.state.examples == plain && irodori.state.text == "Hello.")
        #expect(!voxcpm2.state.managed && voxcpm2.state.model == nil && voxcpm2.state.seed == irodori.state.seed)
        voxcpm2.send(.appeared)
        voxcpm2.send(.setText("Changed"))
        #expect(voxcpm2.state.text == "Changed" && irodori.state.text == "Hello.")
        #expect(defaults.string(forKey: "voxcpm2ModelPath") == "/models/voxcpm2")
        #expect(defaults.string(forKey: "irodoriModelPath") == "/models/irodori")
    }
}
