import FeatureStore
import Foundation
import SpeechCore
import SpeechFeature
import Testing

struct VoiceTranslationFeatureTests {
    let text = " Hello, world. "
    let model = SpeechModel(name: "Test model", directory: URL(filePath: "/test-model"))
    let audio = SpeechAudio(wav: Data([1, 2, 3]), audioSeconds: 1, elapsedSeconds: 2, normalizedText: "Test")
    let kana = "\u{3084}\u{3055}\u{3057}\u{3044}\u{58F0}"

    func ready(voice: String, text: String) -> SpeechState {
        var state = SpeechState()
        state.presented = true
        state.model = model
        state.text = text
        state.voice = voice
        return state
    }

    func withAudio(voice: String) throws -> SpeechState {
        var state = ready(voice: voice, text: text)
        _ = SpeechFeature.reduce(&state, .generate)
        _ = SpeechFeature.reduce(&state, .generated(state.session, audio))
        _ = SpeechFeature.reduce(&state, .prepareExport)
        let ticket = try #require(state.exportTicket)
        _ = SpeechFeature.reduce(&state, .exportDismissed(ticket))
        try #require(state.canPlay && state.message != nil && state.exportTicket != nil)
        return state
    }

    @Test(
        "TRN-01 A voice description with hiragana or katakana is translated",
        arguments: [
            "\u{3084}\u{3055}\u{3057}\u{3044}\u{58F0}", "\u{30CF}\u{30B9}\u{30AD}\u{30FC}",
            "\u{FF8A}\u{FF7D}\u{FF77}\u{FF70}", "A calm \u{3084}\u{3055}\u{3057}\u{3044} voice",
            "\u{3041}", "\u{3096}", "\u{30A1}", "\u{30FA}", "\u{31F0}", "\u{31FF}",
            "\u{FF66}", "\u{FF6F}", "\u{FF71}", "\u{FF9D}",
        ])
    func kanaVoice(voice: String) {
        var state = ready(voice: voice, text: text)
        let commands = SpeechFeature.reduce(&state, .generate)
        #expect(commands == [.run(.generateTranslated(state.session, voice, text, 1708), id: .operation)])
    }

    @Test(
        "TRN-01 A voice description without hiragana or katakana is not translated",
        arguments: [
            "A calm voice", "\u{6E29}\u{67D4}\u{7684}\u{5973}\u{58F0}", "\u{4F4E}\u{97F3}",
            "", " \n", "( )", "\u{FF08}\u{3000}\u{FF09}",
            "\u{30FC}", "\u{30FB}", "\u{30FC}\u{30FB}\u{309B}\u{309C}", "\u{FF70}", "\u{FF65}\u{FF9E}\u{FF9F}",
            "\u{30A0}\u{30FD}",
        ])
    func otherVoice(voice: String) {
        var state = ready(voice: voice, text: text)
        let commands = SpeechFeature.reduce(&state, .generate)
        #expect(commands == [.run(.generate(state.session, state.prompt, 1708), id: .operation)])
    }

    @Test("TRN-01 A kana voice description is not translated for text with nothing to read or with a reference voice")
    func kanaVoiceWithoutTranslation() throws {
        var blank = ready(voice: kana, text: " \u{1C}\n")
        let blankCommands = SpeechFeature.reduce(&blank, .generate)
        #expect(blankCommands == [.run(.generate(blank.session, " \u{1C}\n", 1708), id: .operation)])

        var referenced = SpeechState(
            managedModel: SpeechModel(
                name: "Reference", directory: URL(filePath: "/model"), supportsReferenceAudio: true))
        referenced.presented = true
        referenced.text = text
        referenced.voice = kana
        let reference = try SpeechReferenceAudio(samples: [Float](repeating: 0.25, count: 24_000), source: .recording)
        referenced.reference = reference
        let referencedCommands = SpeechFeature.reduce(&referenced, .generate)
        #expect(
            referencedCommands == [
                .run(.generateReference(referenced.session, "(" + kana + ")" + text, 1708, reference), id: .operation)
            ])
    }

    @Test("TRN-02 A kana voice description asks for a translated generation and changes the state like any generation")
    func translatedGeneration() throws {
        var translated = try withAudio(voice: " (" + kana + ") \n")
        var plain = try withAudio(voice: "A calm voice")
        let commands = SpeechFeature.reduce(&translated, .generate)
        #expect(commands == [.run(.generateTranslated(translated.session, kana, text, 1708), id: .operation)])
        _ = SpeechFeature.reduce(&plain, .generate)
        #expect(translated.phase == .generating && translated.audio == nil && translated.message == nil)
        #expect(translated.exportTicket == nil)
        translated.voice = plain.voice
        #expect(translated == plain)
    }
}
