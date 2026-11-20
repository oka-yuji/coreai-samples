import FeatureStore
import Foundation
import SpeechCore
import SpeechFeature
import Testing

struct VoxCPMReferencePromptTests {
    @Test("VCP-01 A VoxCPM2 reference keeps the saved description and sends only the original text to generation")
    func plainReferencePrompt() throws {
        let reference = try SpeechReferenceAudio(
            samples: Array(repeating: 0.25, count: 80_000), source: .file("voice.wav"), format: .voxcpm2)
        var state = SpeechState(
            managedModel: SpeechModel(
                name: "Reference", directory: URL(filePath: "/model"), supportsReferenceAudio: true))
        state.presented = true
        state.voice = "A gentle voice"
        state.text = "  read this\nexactly  "
        state.seed = "1708"
        state.reference = reference
        #expect(state.prompt == state.text)
        let commands = SpeechFeature.reduce(&state, .generate)
        #expect(state.voice == "A gentle voice")
        #expect(state.reference == reference)
        #expect(
            commands == [.run(.generateReference(1, "  read this\nexactly  ", 1708, reference), id: .operation)])
    }
}
