import FeatureStore
import Foundation
import SpeechCore
import SpeechFeature
import Testing

struct VoxCPMReferencePromptLifecycleTests {
    static let english = "A gentle voice"
    static let kana = String(UnicodeScalar(0x304a)!)
    static let text = "  read this\nexactly  "

    @Test(
        "VCP-01 A selected VoxCPM2 reference keeps the description dormant and ignores stale voice edits",
        arguments: [english, kana])
    func dormantDescription(voice: String) throws {
        var state = ready(voice: voice)
        let reference = try vox()
        try select(reference, into: &state)
        #expect(state.reference == reference && state.voice == voice)
        #expect(state.prompt == Self.text && !state.usesVoiceDescription && !state.canEditVoice)
        #expect(state.canGenerate)
        let before = state
        #expect(SpeechFeature.reduce(&state, .setVoice("stale-field-callback")).isEmpty)
        #expect(state == before)
        let session = state.session
        #expect(
            SpeechFeature.reduce(&state, .generate) == [
                .run(.generateReference(session + 1, Self.text, 1708, reference), id: .operation)
            ])
        #expect(state.voice == voice && state.text == Self.text && state.seed == "1708")
    }

    @Test(
        "VCP-02 Choosing an example during a VoxCPM2 reference changes only the text and clearing restores the description",
        arguments: [english, kana])
    func exampleAndClear(voice: String) throws {
        var state = ready(voice: voice)
        let reference = try vox()
        try select(reference, into: &state)
        _ = SpeechFeature.reduce(&state, .generate)
        let ticket = state.session
        let audio = SpeechAudio(wav: Data([1]), audioSeconds: 1, elapsedSeconds: 1, normalizedText: Self.text)
        _ = SpeechFeature.reduce(&state, .generated(ticket, audio))
        _ = SpeechFeature.reduce(&state, .prepareExport)
        #expect(state.audio == audio && state.exportTicket != nil)
        _ = SpeechFeature.reduce(&state, .exportDismissed(try #require(state.exportTicket)))
        let example = SpeechExample(id: "2", title: "Other", text: " second body ", voice: "different voice")
        #expect(SpeechFeature.reduce(&state, .selectExample(example)).isEmpty)
        #expect(state.text == example.text && state.voice == voice && state.reference == reference)
        #expect(state.audio == nil && state.exportTicket == nil && state.prompt == example.text)
        #expect(SpeechFeature.reduce(&state, .clearReference).isEmpty)
        #expect(state.reference == nil && state.text == example.text && state.voice == voice)
        #expect(state.seed == "1708" && state.usesVoiceDescription && state.canEditVoice)
        let session = state.session
        let resumed = SpeechFeature.reduce(&state, .generate)
        if voice == Self.english {
            #expect(resumed == [.run(.generate(session + 1, "(" + voice + ")" + example.text, 1708), id: .operation)])
        } else {
            #expect(resumed == [.run(.generateTranslated(session + 1, voice, example.text, 1708), id: .operation)])
        }
        #expect(SpeechFeature.reduce(&state, .generated(ticket, audio)).isEmpty)
        #expect(state.audio == nil)
    }

    @Test("VCP-02 The description is editable again after the VoxCPM2 reference is removed")
    func editableAfterClear() throws {
        var state = ready(voice: Self.english)
        try select(try vox(), into: &state)
        _ = SpeechFeature.reduce(&state, .clearReference)
        _ = SpeechFeature.reduce(&state, .setVoice("A bright voice"))
        #expect(state.voice == "A bright voice")
        let example = SpeechExample(id: "3", title: "Next", text: "Next body", voice: "Calm")
        _ = SpeechFeature.reduce(&state, .selectExample(example))
        #expect(state.voice == "Calm" && state.prompt == "(Calm)Next body")
    }

    @Test("VCP-03 Stale selections, cancelled replacements and old results do not change the VoxCPM2 draft")
    func staleNotifications() throws {
        var state = ready(voice: Self.kana)
        let reference = try vox()
        _ = SpeechFeature.reduce(&state, .importReference(state.session, URL(filePath: "/sample.wav")))
        let selection = state.session
        #expect(SpeechFeature.reduce(&state, .referencePrepared(selection - 1, reference)).isEmpty)
        #expect(state.reference == nil)
        _ = SpeechFeature.reduce(&state, .referencePrepared(selection, reference))
        _ = SpeechFeature.reduce(&state, .importReference(state.session, URL(filePath: "/replacement.wav")))
        _ = SpeechFeature.reduce(&state, .failed(state.session, "invalid input"))
        #expect(state.reference == reference && state.referenceMessage == "invalid input")
        _ = SpeechFeature.reduce(&state, .importReference(state.session, URL(filePath: "/cancelled.wav")))
        let cancelled = state.session
        _ = SpeechFeature.reduce(&state, .stop)
        _ = SpeechFeature.reduce(&state, .stopped(state.session))
        let late = try SpeechReferenceAudio(
            samples: Array(repeating: 0.5, count: 80_000), source: .file("late.wav"), format: .voxcpm2)
        #expect(SpeechFeature.reduce(&state, .referencePrepared(cancelled, late)).isEmpty)
        #expect(state.reference == reference && state.voice == Self.kana && state.phase == .idle)
        _ = SpeechFeature.reduce(&state, .generate)
        let generation = state.session
        _ = SpeechFeature.reduce(&state, .stop)
        let audio = SpeechAudio(wav: Data([1]), audioSeconds: 1, elapsedSeconds: 1, normalizedText: Self.text)
        #expect(SpeechFeature.reduce(&state, .generated(generation, audio)).isEmpty)
        #expect(SpeechFeature.reduce(&state, .failed(generation, "late")).isEmpty)
        _ = SpeechFeature.reduce(&state, .stopped(state.session))
        #expect(state.audio == nil && state.message == "Stopped." && state.voice == Self.kana)
        #expect(state.reference == reference && state.text == Self.text)
    }

    @Test(
        "VCP-03 Irodori references and generation without a reference keep their previous effects",
        arguments: [english, kana])
    func unchangedPaths(voice: String) throws {
        var plain = ready(voice: voice)
        #expect(plain.usesVoiceDescription && plain.canEditVoice)
        let session = plain.session
        let commands = SpeechFeature.reduce(&plain, .generate)
        if voice == Self.english {
            #expect(commands == [.run(.generate(session + 1, "(" + voice + ")" + Self.text, 1708), id: .operation)])
        } else {
            #expect(commands == [.run(.generateTranslated(session + 1, voice, Self.text, 1708), id: .operation)])
        }
        var irodori = ready(voice: voice)
        let reference = try SpeechReferenceAudio(samples: Array(repeating: 0.25, count: 24_000), source: .recording)
        try select(reference, into: &irodori)
        #expect(irodori.usesVoiceDescription && irodori.canEditVoice)
        let prompt = "(" + voice + ")" + Self.text
        #expect(
            SpeechFeature.reduce(&irodori, .generate) == [
                .run(.generateReference(irodori.session, prompt, 1708, reference), id: .operation)
            ])
        _ = SpeechFeature.reduce(&irodori, .stop)
        _ = SpeechFeature.reduce(&irodori, .stopped(irodori.session))
        let example = SpeechExample(id: "2", title: "Other", text: "Body", voice: "Calm")
        _ = SpeechFeature.reduce(&irodori, .selectExample(example))
        #expect(irodori.voice == "Calm" && irodori.prompt == "(Calm)Body")
        _ = SpeechFeature.reduce(&irodori, .setVoice("Bright"))
        #expect(irodori.voice == "Bright" && irodori.reference == reference)
    }

    private func ready(voice: String) -> SpeechState {
        var state = SpeechState()
        _ = SpeechFeature.reduce(&state, .appeared)
        state.model = SpeechModel(name: "selected", directory: URL(filePath: "/model"), supportsReferenceAudio: true)
        state.voice = voice
        state.text = Self.text
        state.seed = "1708"
        return state
    }

    private func vox() throws -> SpeechReferenceAudio {
        try SpeechReferenceAudio(
            samples: Array(repeating: 0.25, count: 80_000), source: .file("voice.wav"), format: .voxcpm2)
    }

    private func select(_ reference: SpeechReferenceAudio, into state: inout SpeechState) throws {
        _ = SpeechFeature.reduce(&state, .importReference(state.session, URL(filePath: "/voice.wav")))
        _ = SpeechFeature.reduce(&state, .referencePrepared(state.session, reference))
        try #require(state.reference == reference && state.phase == .idle)
    }
}
