import FeatureStore
import Foundation
import SpeechCore
import SpeechFeature
import Testing

struct SpeechReferenceFeatureTests {
    @Test("REF-UI-01 Recording is explicit and its selected voice is passed to generation")
    func recording() throws {
        var state = ready()
        #expect(SpeechFeature.reduce(&state, .recordReference) == [.run(.recordReference(1), id: .operation)])
        #expect(state.phase == .preparingRecording && !state.canGenerate)
        #expect(SpeechFeature.reduce(&state, .referenceRecordingStarted(1)).isEmpty)
        #expect(state.phase == .recording)
        #expect(
            SpeechFeature.reduce(&state, .finishReferenceRecording) == [
                .run(.finishReferenceRecording(1), id: .recordingControl)
            ])
        let reference = try voice()
        #expect(SpeechFeature.reduce(&state, .referencePrepared(1, reference)).isEmpty)
        #expect(state.reference == reference && state.canGenerate)
        #expect(
            SpeechFeature.reduce(&state, .generate) == [
                .run(.generateReference(2, "Example", 1708, reference), id: .operation)
            ])
    }

    @Test("REF-UI-01 Failed or cancelled replacement keeps the previous voice and rejects stale results")
    func replacement() throws {
        var state = ready()
        let reference = try voice()
        state.reference = reference
        let url = URL(filePath: "/voice.wav")
        #expect(
            SpeechFeature.reduce(&state, .importReference(0, url)) == [.run(.importReference(1, url), id: .operation)])
        #expect(SpeechFeature.reduce(&state, .failed(1, "Cannot read audio")).isEmpty)
        #expect(state.reference == reference && state.referenceMessage == "Cannot read audio")
        #expect(state.canGenerate)
        #expect(SpeechFeature.reduce(&state, .importReference(0, url)).isEmpty)
        _ = SpeechFeature.reduce(&state, .recordReference)
        _ = SpeechFeature.reduce(&state, .stop)
        let stopped = state.session
        #expect(SpeechFeature.reduce(&state, .referencePrepared(stopped - 1, reference)).isEmpty)
        #expect(state.phase == .stopping)
        _ = SpeechFeature.reduce(&state, .stopped(stopped))
        #expect(state.phase == .idle && state.reference == reference)
    }

    @Test("REF-UI-02 Ending the managed session cancels capture and discards picker and recording completions")
    func ending() throws {
        var state = ready()
        state.managed = true
        let reference = try voice()
        state.reference = reference
        _ = SpeechFeature.reduce(&state, .recordReference)
        #expect(SpeechFeature.reduce(&state, .ended) == [.cancelAll, .run(.stop(2), id: .stop)])
        #expect(state.reference == nil)
        _ = SpeechFeature.reduce(&state, .appeared)
        #expect(SpeechFeature.reduce(&state, .referencePrepared(1, reference)).isEmpty)
        #expect(SpeechFeature.reduce(&state, .importReference(0, URL(filePath: "/old.wav"))).isEmpty)
        _ = SpeechFeature.reduce(&state, .stopped(2))
        #expect(SpeechFeature.reduce(&state, .referenceSelectionFailed(0, "Old picker error")).isEmpty)
        #expect(state.referenceMessage == nil && state.canGenerate)
    }

    @Test("REF-UI-01 Microphone refusal still allows a file and a legacy model cannot capture")
    func permissionAndLegacy() throws {
        var state = ready()
        _ = SpeechFeature.reduce(&state, .recordReference)
        _ = SpeechFeature.reduce(&state, .microphoneDenied(1))
        #expect(state.microphoneDenied && state.canSelectReference && state.phase == .idle)
        let selected = SpeechFeature.reduce(&state, .importReference(1, URL(filePath: "/voice.wav")))
        #expect(!selected.isEmpty && state.phase == .importingReference)
        var legacy = ready()
        legacy.model = SpeechModel(name: "Legacy", directory: URL(filePath: "/legacy"))
        #expect(SpeechFeature.reduce(&legacy, .recordReference).isEmpty)
        #expect(legacy.canGenerate)
    }

    private func ready() -> SpeechState {
        var state = SpeechState(
            managedModel: SpeechModel(
                name: "Reference", directory: URL(filePath: "/model"), supportsReferenceAudio: true))
        state.presented = true
        state.text = "Example"
        return state
    }

    private func voice() throws -> SpeechReferenceAudio {
        try SpeechReferenceAudio(samples: [Float](repeating: 0.25, count: 24_000), source: .recording)
    }
}
