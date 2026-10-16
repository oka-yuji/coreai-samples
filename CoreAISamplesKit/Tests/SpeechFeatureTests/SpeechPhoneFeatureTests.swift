import Foundation
import SpeechFeature
import Testing

struct SpeechPhoneFeatureTests {
    @Test("Refreshing or leaving invalidates older folder listings")
    func librarySessions() {
        var state = SpeechLibraryState()
        #expect(SpeechLibraryFeature.reduce(&state, .refresh).isEmpty)
        #expect(SpeechLibraryFeature.reduce(&state, .appeared) == [.run(.load(1), id: .load)])
        _ = SpeechLibraryFeature.reduce(&state, .refresh)
        let current = state.session
        let folder = SpeechFolder(url: URL(filePath: "/models"))
        _ = SpeechLibraryFeature.reduce(&state, .loaded(1, [folder]))
        #expect(state.folders.isEmpty && state.loading)
        _ = SpeechLibraryFeature.reduce(&state, .failed(current, "Unavailable"))
        #expect(state.message == "Unavailable" && !state.loading)
        _ = SpeechLibraryFeature.reduce(&state, .refresh)
        _ = SpeechLibraryFeature.reduce(&state, .loaded(state.session, [folder]))
        #expect(state.folders == [folder])
        #expect(SpeechLibraryFeature.reduce(&state, .ended) == [.cancelAll])
        _ = SpeechLibraryFeature.reduce(&state, .loaded(state.session - 1, []))
        #expect(state.folders == [folder] && !state.presented)
    }

    @Test("A device check runs once and ignores progress from a stopped session")
    func deviceCheckLifetime() {
        var state = SpeechCheckState()
        let request = SpeechCheckRequest(folder: "model", runID: "run-1")
        #expect(SpeechCheckFeature.reduce(&state, .start(request)) == [.run(.run(1, request), id: .run)])
        #expect(SpeechCheckFeature.reduce(&state, .start(request)).isEmpty)
        _ = SpeechCheckFeature.reduce(&state, .progress(1, "Loading"))
        #expect(state.message == "Loading")
        #expect(SpeechCheckFeature.reduce(&state, .stop) == [.cancelAll, .run(.stop(2), id: .stop)])
        _ = SpeechCheckFeature.reduce(&state, .completed(1, "old"))
        _ = SpeechCheckFeature.reduce(&state, .failed(1, "old"))
        _ = SpeechCheckFeature.reduce(&state, .progress(1, "old"))
        #expect(state.phase == .stopping)
        _ = SpeechCheckFeature.reduce(&state, .stopped(2))
        #expect(state.phase == .finished)
        #expect(SpeechCheckFeature.reduce(&state, .start(request)).isEmpty)
    }

    @Test("Current check success and failure finish the diagnostic screen")
    func outcomes() {
        for failure in [false, true] {
            var state = SpeechCheckState()
            _ = SpeechCheckFeature.reduce(&state, .start(.init(folder: "model", runID: "run")))
            _ = SpeechCheckFeature.reduce(&state, failure ? .failed(1, "Error") : .completed(1, "SpeechChecks/run"))
            #expect(state.phase == .finished)
            let finished = state
            _ = SpeechCheckFeature.reduce(&state, .progress(1, "Late"))
            #expect(state == finished)
        }
    }
}
