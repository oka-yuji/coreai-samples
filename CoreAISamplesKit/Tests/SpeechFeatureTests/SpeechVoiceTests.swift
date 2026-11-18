import FeatureStore
import Foundation
import SpeechCore
import SpeechFeature
import Testing

struct SpeechVoiceTests {
    let text = " Hello, world. "
    let model = SpeechModel(name: "Test model", directory: URL(filePath: "/test-model"))
    let audio = SpeechAudio(wav: Data([1, 2, 3]), audioSeconds: 1, elapsedSeconds: 2, normalizedText: "Test")
    let plain = SpeechExample(id: "plain", title: "Plain", text: "Plain text")
    let calm = SpeechExample(id: "calm", title: "Calm", text: "Calm text", voice: "A calm voice")

    func ready(text: String) -> SpeechState {
        var state = SpeechState(examples: [plain, calm])
        state.presented = true
        state.model = model
        state.text = text
        return state
    }

    func withAudio() throws -> SpeechState {
        var state = ready(text: "Test")
        _ = SpeechFeature.reduce(&state, .generate)
        _ = SpeechFeature.reduce(&state, .generated(state.session, audio))
        _ = SpeechFeature.reduce(&state, .prepareExport)
        let ticket = try #require(state.exportTicket)
        _ = SpeechFeature.reduce(&state, .exportDismissed(ticket))
        try #require(state.canPlay && state.message != nil)
        return state
    }

    func decode(_ json: String) throws -> [SpeechExample] {
        try JSONDecoder().decode([SpeechExample].self, from: Data(json.utf8))
    }

    @Test(
        "VIN-01 An empty, blank or parentheses-only voice description leaves the prompt equal to the text",
        arguments: ["", " ", "\n\t", "\u{3000}", "()", "( )", "\u{FF08}\u{FF09}", " (\u{FF08} ) \u{FF09}"])
    func emptyVoice(voice: String) {
        var state = ready(text: text)
        _ = SpeechFeature.reduce(&state, .setVoice(voice))
        #expect(state.voice == voice)
        #expect(state.prompt == text)
        #expect(SpeechExample(id: "a", title: "A", text: text, voice: voice).prompt == text)
    }

    @Test(
        "VIN-02 The voice description is trimmed, cleared of parentheses and put in parentheses before the text",
        arguments: [
            ("A calm voice", "(A calm voice)"),
            ("  A calm voice \n", "(A calm voice)"),
            ("(A calm voice)", "(A calm voice)"),
            ("\u{FF08}A calm voice\u{FF09}", "(A calm voice)"),
            (" ( A calm voice ) ", "(A calm voice)"),
            ("A (young) woman", "(A young woman)"),
            ("A calm\nvoice", "(A calm\nvoice)"),
            ("A  calm voice", "(A  calm voice)"),
            ("A calm (\u{0301}voice", "(A calm \u{0301}voice)"),
        ])
    func voicePrefix(voice: String, prefix: String) {
        var state = ready(text: text)
        _ = SpeechFeature.reduce(&state, .setVoice(voice))
        #expect(state.prompt == prefix + text)
        #expect(SpeechExample(id: "a", title: "A", text: text, voice: voice).prompt == prefix + text)
    }

    @Test("VIN-02 Parentheses in the text stay unchanged after the voice description")
    func textParentheses() {
        var state = ready(text: "(aside) Hello")
        _ = SpeechFeature.reduce(&state, .setVoice("A calm voice"))
        #expect(state.prompt == "(A calm voice)(aside) Hello")
        let example = SpeechExample(id: "a", title: "A", text: "(aside) Hello", voice: "A calm voice")
        #expect(example.prompt == "(A calm voice)(aside) Hello")
    }

    @Test("VIN-03 Generation passes the voice description and the text to the runtime as one prompt")
    func generation() throws {
        var state = ready(text: "Test")
        _ = SpeechFeature.reduce(&state, .setVoice("A calm voice"))
        let commands = SpeechFeature.reduce(&state, .generate)
        #expect(commands == [.run(.generate(state.session, "(A calm voice)Test", 1708), id: .operation)])

        var referenced = SpeechState(
            managedModel: SpeechModel(
                name: "Reference", directory: URL(filePath: "/model"), supportsReferenceAudio: true))
        referenced.presented = true
        referenced.text = "Test"
        let reference = try SpeechReferenceAudio(samples: [Float](repeating: 0.25, count: 24_000), source: .recording)
        referenced.reference = reference
        _ = SpeechFeature.reduce(&referenced, .setVoice("A calm voice"))
        let referencedCommands = SpeechFeature.reduce(&referenced, .generate)
        #expect(
            referencedCommands == [
                .run(.generateReference(referenced.session, "(A calm voice)Test", 1708, reference), id: .operation)
            ])
    }

    @Test("VIN-03 Changing the voice description discards the generated audio, its message and its save ticket")
    func voiceDiscardsAudio() throws {
        var state = try withAudio()
        let session = state.session
        #expect(SpeechFeature.reduce(&state, .setVoice("A calm voice")).isEmpty)
        #expect(state.voice == "A calm voice")
        #expect(state.audio == nil && state.message == nil && state.exportTicket == nil)
        #expect(state.session == session + 1)
    }

    @Test("VIN-03 Voice and example edits are ignored while generating and while the screen is not shown")
    func editsWhileNotEditable() throws {
        var state = ready(text: "Test")
        _ = SpeechFeature.reduce(&state, .generate)
        try #require(state.phase == .generating)
        let running = state
        #expect(SpeechFeature.reduce(&state, .setVoice("A calm voice")).isEmpty)
        #expect(SpeechFeature.reduce(&state, .selectExample(calm)).isEmpty)
        #expect(state == running)

        var hidden = SpeechState(examples: [plain, calm])
        hidden.model = model
        try #require(!hidden.presented)
        let initial = hidden
        #expect(SpeechFeature.reduce(&hidden, .setVoice("A calm voice")).isEmpty)
        #expect(SpeechFeature.reduce(&hidden, .selectExample(calm)).isEmpty)
        #expect(hidden == initial)
    }

    @Test("VIN-03 A voice description does not allow generation when the text is blank")
    func blankTextWithVoice() {
        var state = ready(text: "Test")
        _ = SpeechFeature.reduce(&state, .setVoice("A calm voice"))
        _ = SpeechFeature.reduce(&state, .setText(" \n\t"))
        #expect(state.prompt == "(A calm voice) \n\t")
        #expect(!state.canGenerate)
        #expect(SpeechFeature.reduce(&state, .generate).isEmpty)
    }

    @Test("VIN-04 The initial text and voice description come from the first example")
    func initialVoice() {
        let plainFirst = SpeechState(examples: [plain, calm])
        #expect(plainFirst.text == plain.text && plainFirst.voice.isEmpty)
        let calmFirst = SpeechState(examples: [calm, plain])
        #expect(calmFirst.text == calm.text && calmFirst.voice == "A calm voice")
        let managed = SpeechState(managedModel: model, examples: [calm, plain])
        #expect(managed.text == calm.text && managed.voice == "A calm voice")
    }

    @Test("VIN-04 Selecting an example replaces the text and the voice description and discards audio once")
    func exampleSelection() throws {
        var state = try withAudio()
        let session = state.session
        #expect(SpeechFeature.reduce(&state, .selectExample(calm)).isEmpty)
        #expect(state.text == calm.text && state.voice == "A calm voice")
        #expect(state.audio == nil && state.message == nil && state.exportTicket == nil)
        #expect(state.session == session + 1)
        _ = SpeechFeature.reduce(&state, .selectExample(plain))
        #expect(state.text == plain.text && state.voice.isEmpty)
    }

    @Test("VIN-04 A managed screen keeps the voice description of the previous state")
    func retainedVoice() {
        var previous = SpeechState(managedModel: model, examples: [plain, calm])
        previous.voice = "A bright voice"
        let resumed = SpeechState(managedModel: model, examples: [calm, plain], retaining: previous)
        #expect(resumed.voice == "A bright voice")
        previous.voice = ""
        let cleared = SpeechState(managedModel: model, examples: [calm, plain], retaining: previous)
        #expect(cleared.voice.isEmpty)
    }

    @Test("VIN-05 Examples decode with or without a voice description and survive encoding unchanged")
    func exampleDecoding() throws {
        let plainExamples = try decode(#"[{"id":"a","title":"A","text":"Hello"}]"#)
        #expect(plainExamples == [SpeechExample(id: "a", title: "A", text: "Hello")])
        #expect(plainExamples.map(\.voice) == [nil])
        let calmExamples = try decode(#"[{"id":"a","title":"A","text":"Hello","voice":"A calm voice"}]"#)
        #expect(calmExamples == [SpeechExample(id: "a", title: "A", text: "Hello", voice: "A calm voice")])
        for examples in [plainExamples, calmExamples] {
            let encoded = try JSONEncoder().encode(examples)
            #expect(try JSONDecoder().decode([SpeechExample].self, from: encoded) == examples)
        }
    }

    @Test("VIN-05 Without a voice description generation passes the text unchanged")
    func plainGeneration() throws {
        let examples = try decode(#"[{"id":"a","title":"A","text":"Hello"}]"#)
        var state = SpeechState(managedModel: model, examples: examples)
        _ = SpeechFeature.reduce(&state, .appeared)
        #expect(state.voice.isEmpty && state.prompt == "Hello")
        let commands = SpeechFeature.reduce(&state, .generate)
        #expect(commands == [.run(.generate(state.session, "Hello", 1708), id: .operation)])
    }
}
