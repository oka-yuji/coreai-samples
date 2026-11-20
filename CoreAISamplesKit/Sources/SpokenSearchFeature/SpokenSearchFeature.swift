import FeatureStore
import Foundation

public enum SpokenSearchFeature: Feature {
    public typealias State = SpokenSearchState
    public typealias Action = SpokenSearchAction
    public typealias Effect = SpokenSearchEffect
    public typealias EffectID = SpokenSearchEffectID
    public typealias Output = Command<Effect, EffectID>

    public static func reduce(_ state: inout State, _ action: Action) -> [Output] {
        switch action {
        case .appeared:
            state.isActive = true
        case .disappeared:
            guard state.isActive else { return [] }
            reset(&state)
            state.isActive = false
            return [.cancelAll, .run(.clear(state.generation), id: .cleanup)]
        case .cancel:
            guard state.isActive, state.isWorking else { return [] }
            reset(&state)
            return [.cancelAll, .run(.clear(state.generation), id: .cleanup)]
        case .fileSelected(let url, let generation):
            guard state.isActive, state.generation == generation else { return [] }
            reset(&state)
            state.filename = url.lastPathComponent
            state.isTranscribing = true
            return [.cancelAll, .run(.open(state.generation, url), id: .recognition)]
        case .fileSelectionFailed(let generation):
            guard state.isActive, state.generation == generation else { return [] }
            state.notice = .unreadableFile
        case .queryChanged(let query):
            guard state.isActive, state.query != query else { return [] }
            state.query = query
            state.hits = []
            state.hasSearched = false
            state.isSearching = false
            return [.cancel(.search)] + stopPlaying(&state)
        case .search:
            let query = state.query.trimmingCharacters(in: .whitespacesAndNewlines)
            guard state.isActive, state.hasTranscript, !state.isWorking, !query.isEmpty else { return [] }
            let stop = stopPlaying(&state)
            state.isSearching = true
            state.hits = []
            state.hasSearched = false
            state.notice = nil
            return stop + [.run(.find(state.generation, query), id: .search)]
        case .play(let id):
            guard state.isActive, !state.isWorking, let hit = state.hits.first(where: { $0.id == id }) else {
                return []
            }
            state.playingID = id
            state.notice = nil
            return [.run(.play(state.generation, hit), id: .player)]
        case .stopPlayback:
            return stopPlaying(&state)
        case .recognized(let generation):
            guard state.isActive, state.generation == generation, state.isTranscribing else { return [] }
            state.isTranscribing = false
            state.hasTranscript = true
        case .found(let generation, let hits):
            guard state.isActive, state.generation == generation, state.isSearching else { return [] }
            state.isSearching = false
            state.hasSearched = true
            state.hits = hits
        case .playbackEnded(let generation, let id):
            guard state.generation == generation, state.playingID == id else { return [] }
            state.playingID = nil
        case .failed(let generation, let notice):
            guard state.isActive, state.generation == generation else { return [] }
            state.isTranscribing = false
            state.isSearching = false
            state.playingID = nil
            state.notice = notice
        }
        return []
    }

    private static func stopPlaying(_ state: inout State) -> [Output] {
        guard state.playingID != nil else { return [] }
        state.playingID = nil
        return [.run(.stopPlayback(state.generation), id: .player)]
    }

    private static func reset(_ state: inout State) {
        let generation = state.generation + 1
        let query = state.query
        state = State()
        state.isActive = true
        state.generation = generation
        state.query = query
    }
}
