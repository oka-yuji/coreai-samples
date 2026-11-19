import FeatureStore
import Foundation
import SpeechCore
import SpeechFeature
import Testing

@testable import SpeechRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct VoiceTranslationRunnerTests {
    let voice =
        "\u{82E5}\u{3044}\u{5973}\u{6027}\u{3001}\u{3084}\u{3055}\u{3057}\u{304F}"
        + "\u{843D}\u{3061}\u{7740}\u{3044}\u{305F}\u{58F0}"
    let text = " (aside) Hello, world. "

    func ready(voice: String) -> SpeechState {
        var state = SpeechState()
        state.presented = true
        state.model = SpeechModel(name: "Fixture", directory: URL(filePath: "/model"))
        state.text = text
        state.voice = voice
        return state
    }

    func generate(_ state: SpeechState, on service: FakeVoiceTranslationService) async throws -> Store<SpeechFeature> {
        let runner = SpeechRunner(service: service)
        let finished = TestSignal()
        let store = Store<SpeechFeature>(initialState: state) { effect, send in
            await runner.run(effect, send: send)
            finished.fire()
        }
        store.send(.generate)
        try #require(store.state.phase == .generating)
        try await finished.wait()
        return store
    }

    @Test(
        "TRN-02 The runner puts the cleaned translation in parentheses before the unchanged text",
        arguments: [
            ("A young woman, with a gentle and calm voice", "(A young woman, with a gentle and calm voice)"),
            ("  (A young woman, gentle voice) \n", "(A young woman, gentle voice)"),
            ("\u{FF08}A young woman, gentle voice\u{FF09}", "(A young woman, gentle voice)"),
        ])
    func translatedPrompt(translation: String, prefix: String) async {
        let service = FakeVoiceTranslationService(translation: translation)
        service.gate.open()
        var actions: [SpeechAction] = []
        await SpeechRunner(service: service).run(.generateTranslated(7, voice, text, 42)) { actions.append($0) }
        #expect(service.voices == [voice])
        #expect(service.texts == [prefix + text])
        #expect(service.texts.first?.hasSuffix(text) == true)
        #expect(service.seeds == [42])
        #expect(actions == [.generated(7, service.audio)])
    }

    @Test("TRN-03 Voice descriptions without kana and the examples reach the model as before without a translation")
    func untranslatedPrompts() async throws {
        let file = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "CoreAISamples/Demos/Speech/Resources/voxcpm2-speech-examples.json")
        let examples = try JSONDecoder().decode([SpeechExample].self, from: Data(contentsOf: file))
        try #require(examples.map(\.id) == ["plain", "calm", "steady"])
        let chinese = "\u{6E29}\u{67D4}\u{7684}\u{5973}\u{58F0}"
        let kanji = "\u{4F4E}\u{97F3}"
        var cases: [(state: SpeechState, prompt: String)] = [
            (ready(voice: "A calm voice"), "(A calm voice)" + text),
            (ready(voice: chinese), "(" + chinese + ")" + text),
            (ready(voice: kanji), "(" + kanji + ")" + text),
            (ready(voice: ""), text),
        ]
        for example in examples {
            var state = ready(voice: "")
            _ = SpeechFeature.reduce(&state, .selectExample(example))
            let prompt = (example.voice.map { "(" + $0 + ")" } ?? "") + example.text
            #expect(example.prompt == prompt)
            cases.append((state, prompt))
        }
        for (state, prompt) in cases {
            var next = state
            let commands = SpeechFeature.reduce(&next, .generate)
            #expect(commands == [.run(.generate(next.session, prompt, 1708), id: .operation)])
            let service = FakeVoiceTranslationService(translation: "Unused")
            service.gate.open()
            let store = try await generate(state, on: service)
            #expect(service.voices.isEmpty)
            #expect(service.texts == [prompt])
            #expect(store.state.phase == .idle && store.state.audio == service.audio)
        }
    }

    @Test(
        "TRN-04 A failed or empty translation does not generate, shows the error and returns to idle",
        arguments: [nil, "", " ( ) \n", "\u{FF08}\u{3000}\u{FF09}"] as [String?])
    func failedTranslation(translation: String?) async throws {
        let service = FakeVoiceTranslationService(translation: translation)
        service.gate.open()
        let store = try await generate(ready(voice: voice), on: service)
        #expect(service.voices == [voice])
        #expect(service.texts.isEmpty)
        #expect(store.state.phase == .idle && store.state.audio == nil)
        #expect(store.state.message == SpeechError.voiceNotTranslated.errorDescription)
        #expect(
            SpeechError.voiceNotTranslated.errorDescription
                == "The voice description could not be translated to English. "
                + "Download Japanese and English as translation languages in System Settings, "
                + "or write the description in English.")
    }

    @Test("TRN-05 Stopping during the translation does not generate")
    func stopDuringTranslation() async throws {
        let service = FakeVoiceTranslationService(translation: "A calm voice")
        let runner = SpeechRunner(service: service)
        let generation = TestSignal()
        let stopped = TestSignal()
        let store = Store<SpeechFeature>(initialState: ready(voice: voice)) { effect, send in
            await runner.run(effect, send: send)
            if case .stop = effect { stopped.fire() } else { generation.fire() }
        }
        defer { service.gate.open() }
        store.send(.generate)
        try await service.started.wait()
        store.send(.stop)
        try await stopped.wait()
        #expect(store.state.phase == .idle && store.state.message == "Stopped.")
        service.gate.open()
        try await generation.wait()
        #expect(service.voices == [voice])
        #expect(service.texts.isEmpty)
        #expect(store.state.phase == .idle && store.state.audio == nil)
    }

    @Test(
        "TRN-06 The translation on this Mac returns a voice description that is not empty and has no kana",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_VOICE_TRANSLATION_TEST"] == "1"))
    func liveTranslation() async throws {
        let service = LiveSpeechService(engine: FakeSpeechEngine(), playback: AudioPlayback(), defaults: nil)
        let translation = try await service.translate(voice: voice)
        print("TRN-06 translation: \(translation)")
        #expect(!translation.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        var state = ready(voice: translation)
        let commands = SpeechFeature.reduce(&state, .generate)
        #expect(commands == [.run(.generate(state.session, state.prompt, 1708), id: .operation)])
        #expect(state.prompt != text)
    }
}
