import Foundation
import SpokenSearchFeature
import Testing

struct SpokenModeTests {
    @Test("Model readiness preserves word matches and only the previous topic results become stale")
    func modelsAndModes() throws {
        var state = SpokenSearchState()
        _ = SpokenSearchFeature.reduce(&state, .topicSourceChanged("first"))
        #expect(state.hasTopicModel)
        _ = SpokenSearchFeature.reduce(&state, .appeared)
        _ = SpokenSearchFeature.reduce(&state, .fileSelected(URL(filePath: "/rain.wav"), generation: state.generation))
        let file = state.generation
        _ = SpokenSearchFeature.reduce(&state, .recognized(file))
        _ = SpokenSearchFeature.reduce(&state, .queryChanged("coat"))
        _ = SpokenSearchFeature.reduce(&state, .search)
        let match = SpokenHit(id: 10, text: "raincoat", start: 3, end: 4)
        _ = SpokenSearchFeature.reduce(&state, .found(file, [match]))
        _ = SpokenSearchFeature.reduce(&state, .topicSourceChanged(nil))
        #expect(state.hits == [match] && state.hasTranscript && !state.hasTopicModel)
        _ = SpokenSearchFeature.reduce(&state, .modeChanged(.topic))
        #expect(SpokenSearchFeature.reduce(&state, .search).isEmpty)
        #expect(state.notice == .topicUnavailable)
        _ = SpokenSearchFeature.reduce(&state, .topicSourceChanged("second"))
        let request = SpokenSearchFeature.reduce(&state, .search)
        #expect(request == [.run(.findTopic(file, "coat", "second"), id: .search)])
        _ = SpokenSearchFeature.reduce(&state, .topicSourceChanged("third"))
        _ = SpokenSearchFeature.reduce(&state, .topicFound(file, "second", [match]))
        #expect(state.isWorking && SpokenSearchFeature.reduce(&state, .search).isEmpty)
        _ = SpokenSearchFeature.reduce(&state, .topicStopped(file))
        #expect(!state.isWorking)
        #expect(state.hits.isEmpty && !state.hasSearched && state.hasTranscript)
        #expect(state.filename == "rain.wav" && state.query == "coat")
    }
}
