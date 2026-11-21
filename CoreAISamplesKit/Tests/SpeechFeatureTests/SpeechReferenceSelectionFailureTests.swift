import Foundation
import SpeechCore
import SpeechFeature
import Testing

struct SpeechReferenceSelectionFailureTests {
    @Test("A current file selection error replaces the microphone remedy and preserves the selected voice and inputs")
    func currentFailure() throws {
        var state = try denied()
        let previous = state
        let error = "The file chooser could not finish."
        let commands = SpeechFeature.reduce(&state, .referenceSelectionFailed(previous.session, error))
        #expect(commands.isEmpty && state.phase == .idle && state.session == previous.session)
        #expect(state.referenceMessage == error)
        #expect(!state.microphoneDenied)
        let keptReference = state.reference == previous.reference
        #expect(keptReference && state.canSelectReference && state.canGenerate)
        #expect(state.text == previous.text && state.voice == previous.voice && state.seed == previous.seed)
    }

    @Test("A stale, busy or dismissed file chooser cannot clear the current microphone remedy")
    func rejectsOldFailures() throws {
        let initial = try denied()
        var state = initial
        #expect(
            SpeechFeature.reduce(&state, .referenceSelectionFailed(initial.session - 1, "Stale chooser")).isEmpty)
        let unchanged = state == initial
        #expect(unchanged)

        _ = SpeechFeature.reduce(&state, .playReference)
        try #require(state.phase == .playing && state.microphoneDenied)
        let playing = state
        #expect(SpeechFeature.reduce(&state, .referenceSelectionFailed(playing.session, "Busy chooser")).isEmpty)
        let stillPlaying = state == playing
        #expect(stillPlaying)

        _ = SpeechFeature.reduce(&state, .ended)
        try #require(!state.presented)
        let dismissed = state
        #expect(SpeechFeature.reduce(&state, .referenceSelectionFailed(dismissed.session, "Dismissed chooser")).isEmpty)
        let stillDismissed = state == dismissed
        #expect(stillDismissed)
    }

    private func denied() throws -> SpeechState {
        let file = URL(filePath: "/synthetic/selected.wav")
        let model = SpeechModel(
            name: "Reference", directory: URL(filePath: "/synthetic/voice-model"), supportsReferenceAudio: true)
        let reference = try SpeechReferenceAudio(
            samples: Array(repeating: 0.25, count: 80_000), source: .file(file.lastPathComponent), format: .voxcpm2)
        var state = SpeechState()
        _ = SpeechFeature.reduce(&state, .appeared)
        _ = SpeechFeature.reduce(&state, .openBundle(model.directory.path))
        _ = SpeechFeature.reduce(&state, .opened(state.session, model))
        _ = SpeechFeature.reduce(&state, .setText("  A separate sentence.  "))
        _ = SpeechFeature.reduce(&state, .setVoice("A saved description"))
        _ = SpeechFeature.reduce(&state, .setSeed("731"))
        _ = SpeechFeature.reduce(&state, .importReference(state.session, file))
        _ = SpeechFeature.reduce(&state, .referencePrepared(state.session, reference))
        _ = SpeechFeature.reduce(&state, .recordReference)
        _ = SpeechFeature.reduce(&state, .microphoneDenied(state.session))
        try #require(
            state.microphoneDenied && state.referenceMessage == SpeechReferenceError.permissionDenied.errorDescription)
        try #require(state.reference != nil && state.canGenerate && state.canSelectReference)
        return state
    }
}
