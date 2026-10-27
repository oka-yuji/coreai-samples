import FeatureStore
import Foundation
import SpeechCore
import SpeechFeature
import Testing

struct SpeechFeatureTests {
    let model = SpeechModel(name: "Test model", directory: URL(filePath: "/test-model"))
    let audio = SpeechAudio(wav: Data([1, 2, 3]), audioSeconds: 1, elapsedSeconds: 2, normalizedText: "Test")

    func ready() -> SpeechState {
        var state = SpeechState()
        state.presented = true
        state.model = model
        state.text = "Test"
        return state
    }

    @Test("Generation requires a model, nonempty text, a valid seed and an idle screen")
    func inputGuards() {
        var state = SpeechState()
        #expect(SpeechFeature.reduce(&state, .generate).isEmpty)
        _ = SpeechFeature.reduce(&state, .appeared)
        _ = SpeechFeature.reduce(&state, .setText("Test"))
        #expect(!state.canGenerate)
        state.model = model
        _ = SpeechFeature.reduce(&state, .setText(" \n\t"))
        #expect(SpeechFeature.reduce(&state, .generate).isEmpty)
        _ = SpeechFeature.reduce(&state, .setText("Test"))
        for seed in ["", "-1", "1.5", "18446744073709551616"] {
            _ = SpeechFeature.reduce(&state, .setSeed(seed))
            #expect(SpeechFeature.reduce(&state, .generate).isEmpty)
        }
        _ = SpeechFeature.reduce(&state, .setSeed("18446744073709551615"))
        let commands = SpeechFeature.reduce(&state, .generate)
        #expect(commands == [.run(.generate(state.session, "Test", UInt64.max), id: .operation)])
        let running = state
        #expect(SpeechFeature.reduce(&state, .generate).isEmpty)
        #expect(SpeechFeature.reduce(&state, .setText("Changed")).isEmpty)
        #expect(SpeechFeature.reduce(&state, .openBundle("/other")).isEmpty)
        #expect(state == running)
    }

    @Test("Model selection and opening failures have separate sessions")
    func modelSelection() {
        var state = ready()
        state.audio = audio
        let commands = SpeechFeature.reduce(&state, .openBundle("/test-model"))
        #expect(commands == [.run(.open(state.session, "/test-model"), id: .operation)])
        #expect(state.model == nil && state.audio == nil)
        _ = SpeechFeature.reduce(&state, .opened(state.session - 1, model))
        #expect(state.phase == .opening)
        _ = SpeechFeature.reduce(&state, .failed(state.session, "Invalid bundle"))
        #expect(state.phase == .idle && !state.canGenerate)
        _ = SpeechFeature.reduce(&state, .openBundle("/test-model"))
        _ = SpeechFeature.reduce(&state, .opened(state.session, model))
        #expect(state.canGenerate && state.lastPath == "/test-model")
    }

    @Test("Stop waits for acknowledgement and rejects old generation results")
    func stopGeneration() {
        var state = ready()
        _ = SpeechFeature.reduce(&state, .generate)
        let generation = state.session
        let commands = SpeechFeature.reduce(&state, .stop)
        #expect(commands == [.cancel(.operation), .run(.stop(state.session), id: .stop)])
        _ = SpeechFeature.reduce(&state, .generated(generation, audio))
        _ = SpeechFeature.reduce(&state, .failed(generation, "Old failure"))
        _ = SpeechFeature.reduce(&state, .stopped(generation))
        #expect(state.phase == .stopping && state.audio == nil)
        #expect(SpeechFeature.reduce(&state, .generate).isEmpty)
        _ = SpeechFeature.reduce(&state, .stopped(state.session))
        #expect(state.canGenerate)
        _ = SpeechFeature.reduce(&state, .generate)
        _ = SpeechFeature.reduce(&state, .generated(generation, audio))
        #expect(state.audio == nil)
        _ = SpeechFeature.reduce(&state, .generated(state.session, audio))
        #expect(state.canPlay && state.audio == audio)
    }

    @Test("Leaving and returning waits for release before allowing another operation")
    func screenLifetime() {
        var state = ready()
        _ = SpeechFeature.reduce(&state, .generate)
        let old = state.session
        #expect(SpeechFeature.reduce(&state, .ended) == [.cancelAll, .run(.stop(old + 1), id: .stop)])
        _ = SpeechFeature.reduce(&state, .generated(old, audio))
        #expect(!state.presented && state.audio == nil)
        _ = SpeechFeature.reduce(&state, .appeared)
        #expect(!state.canGenerate)
        _ = SpeechFeature.reduce(&state, .stopped(state.session))
        #expect(state.canGenerate)
        _ = SpeechFeature.reduce(&state, .ended)
        _ = SpeechFeature.reduce(&state, .stopped(state.session))
        #expect(!state.presented && state.phase == .idle)
    }

    @Test("Playback failure retains the audio while edits invalidate it")
    func playbackAndFailure() {
        var state = ready()
        state.audio = audio
        #expect(SpeechFeature.reduce(&state, .play) == [.run(.play(1, audio.wav), id: .operation)])
        _ = SpeechFeature.reduce(&state, .playbackFinished(0))
        #expect(state.phase == .playing)
        _ = SpeechFeature.reduce(&state, .failed(1, "Playback failed"))
        #expect(state.canPlay && state.message == "Playback failed")
        _ = SpeechFeature.reduce(&state, .play)
        _ = SpeechFeature.reduce(&state, .playbackFinished(state.session))
        #expect(state.canPlay)
        _ = SpeechFeature.reduce(&state, .setSeed("18"))
        #expect(state.audio == nil && !state.canPlay)
        _ = SpeechFeature.reduce(&state, .generate)
        _ = SpeechFeature.reduce(&state, .failed(state.session, "Generation failed"))
        #expect(state.canGenerate && state.audio == nil && state.message == "Generation failed")
    }

    @Test("Dismissed save panels accept their result until a new session invalidates them")
    func exportSessions() throws {
        var state = ready()
        state.audio = audio
        _ = SpeechFeature.reduce(&state, .prepareExport)
        let first = try #require(state.exportTicket)
        #expect(!state.editable && !state.canPlay)
        _ = SpeechFeature.reduce(&state, .exportDismissed(first))
        #expect(state.editable)
        _ = SpeechFeature.reduce(&state, .exportFinished(first, "Disk full"))
        #expect(state.message == "Disk full" && state.exportTicket == nil)
        _ = SpeechFeature.reduce(&state, .prepareExport)
        let second = try #require(state.exportTicket)
        _ = SpeechFeature.reduce(&state, .exportDismissed(second))
        _ = SpeechFeature.reduce(&state, .setText("New text"))
        _ = SpeechFeature.reduce(&state, .exportFinished(second, nil))
        #expect(state.message == nil && state.audio == nil)
    }

    @Test("An old save completion cannot dismiss a newer panel")
    func exportTickets() throws {
        var state = ready()
        state.audio = audio
        _ = SpeechFeature.reduce(&state, .prepareExport)
        let old = try #require(state.exportTicket)
        _ = SpeechFeature.reduce(&state, .exportDismissed(old))
        _ = SpeechFeature.reduce(&state, .prepareExport)
        let new = try #require(state.exportTicket)
        #expect(new.id != old.id)
        _ = SpeechFeature.reduce(&state, .exportFinished(old, nil))
        #expect(state.exportVisible && state.exportTicket == new)
        _ = SpeechFeature.reduce(&state, .exportFinished(new, nil))
        #expect(state.exportTicket == nil && state.message == "WAV file saved.")
    }
}

extension SpeechFeatureTests {
    @Test(
        "IR-SESSION-02 A managed speech screen keeps Models selection and invalidates old audio and save tickets on exit"
    )
    func managedLifetime() throws {
        var state = SpeechState(managedModel: model)
        #expect(SpeechFeature.reduce(&state, .appeared).isEmpty)
        _ = SpeechFeature.reduce(&state, .setText("Test"))
        #expect(SpeechFeature.reduce(&state, .openBundle("/other")).isEmpty)
        #expect(state.model == model)
        _ = SpeechFeature.reduce(&state, .generate)
        let generation = state.session
        _ = SpeechFeature.reduce(&state, .generated(generation, audio))
        _ = SpeechFeature.reduce(&state, .prepareExport)
        let ticket = try #require(state.exportTicket)
        let commands = SpeechFeature.reduce(&state, .ended)
        #expect(commands == [.cancelAll, .run(.stop(state.session), id: .stop)])
        #expect(state.audio == nil && state.message == nil && state.exportTicket == nil && !state.canPlay)
        let stopped = state
        #expect(SpeechFeature.reduce(&state, .generated(generation, audio)).isEmpty)
        #expect(SpeechFeature.reduce(&state, .exportFinished(ticket, nil)).isEmpty)
        #expect(state == stopped)
        let resumed = SpeechState(managedModel: model, retaining: state)
        #expect(resumed.text == "Test" && resumed.audio == nil && resumed.exportTicket == nil)
    }
}
