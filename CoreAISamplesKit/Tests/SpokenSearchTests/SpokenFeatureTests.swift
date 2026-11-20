import Foundation
import SpokenSearchFeature
import Testing

struct SpokenFeatureTests {
    @Test("An earlier file or search cannot publish results after a replacement, stop, or departure")
    func staleEvents() {
        var state = SpokenSearchState()
        let file = URL(filePath: "/first.wav")
        _ = SpokenSearchFeature.reduce(&state, .appeared)
        _ = SpokenSearchFeature.reduce(&state, .fileSelected(file, generation: state.generation))
        let first = state.generation
        _ = SpokenSearchFeature.reduce(&state, .fileSelected(URL(filePath: "/second.wav"), generation: first))
        let second = state.generation
        #expect(second > first)
        _ = SpokenSearchFeature.reduce(&state, .recognized(first))
        #expect(state.isWorking && !state.hasTranscript)
        _ = SpokenSearchFeature.reduce(&state, .recognized(second))
        _ = SpokenSearchFeature.reduce(&state, .queryChanged("forecast"))
        _ = SpokenSearchFeature.reduce(&state, .search)
        _ = SpokenSearchFeature.reduce(&state, .cancel)
        _ = SpokenSearchFeature.reduce(&state, .found(second, []))
        #expect(!state.hasSearched && !state.isWorking)
        _ = SpokenSearchFeature.reduce(&state, .disappeared)
        _ = SpokenSearchFeature.reduce(&state, .appeared)
        #expect(SpokenSearchFeature.reduce(&state, .fileSelected(file, generation: second)).isEmpty)
        #expect(state.filename == nil)
        _ = SpokenSearchFeature.reduce(&state, .fileSelected(file, generation: state.generation))
        #expect(state.isWorking)
    }

    @Test("Playback uses the selected result and editing the query stops the old recording")
    func playback() {
        var state = SpokenSearchState()
        _ = SpokenSearchFeature.reduce(&state, .appeared)
        _ = SpokenSearchFeature.reduce(
            &state, .fileSelected(URL(filePath: "/record.wav"), generation: state.generation))
        let ticket = state.generation
        _ = SpokenSearchFeature.reduce(&state, .recognized(ticket))
        _ = SpokenSearchFeature.reduce(&state, .queryChanged("weather"))
        _ = SpokenSearchFeature.reduce(&state, .search)
        let hit = SpokenHit(id: 8, text: "the weather is good", start: 3.2, end: 3.8)
        _ = SpokenSearchFeature.reduce(&state, .found(ticket, [hit]))
        #expect(SpokenSearchFeature.reduce(&state, .play(0)).isEmpty)
        #expect(SpokenSearchFeature.reduce(&state, .play(8)) == [.run(.play(ticket, hit), id: .player)])
        let commands = SpokenSearchFeature.reduce(&state, .queryChanged("trains"))
        #expect(commands.contains(.run(.stopPlayback(ticket), id: .player)))
        #expect(state.hits.isEmpty && state.playingID == nil && !state.hasSearched)
        _ = SpokenSearchFeature.reduce(&state, .search)
        _ = SpokenSearchFeature.reduce(&state, .found(ticket, []))
        #expect(state.hasSearched && state.hits.isEmpty)
    }
}
