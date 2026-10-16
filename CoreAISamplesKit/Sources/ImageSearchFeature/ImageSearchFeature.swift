import FeatureStore
import Foundation

public struct SearchPhoto: Equatable, Sendable, Identifiable {
    public let url: URL
    public let thumbnail: Data
    public var id: URL { url }
    public init(url: URL, thumbnail: Data) {
        self.url = url
        self.thumbnail = thumbnail
    }
}

public struct ImageSearchMatch: Equatable, Sendable, Identifiable {
    public let photo: SearchPhoto
    public let similarity: Double
    public var id: URL { photo.id }
    public init(photo: SearchPhoto, similarity: Double) {
        self.photo = photo
        self.similarity = similarity
    }
}

public struct ImageSearchFailure: Equatable, Sendable {
    public let message: String
    public let detail: String
    public init(message: String, detail: String = "") {
        self.message = message
        self.detail = detail
    }
}

public enum ImageSearchPhase: Equatable, Sendable {
    case idle, loading, indexing, searching, stopping
}

public struct ImageSearchState: Equatable, Sendable {
    public static let minimumSimilarity = 0.67
    public var modelFolder: URL?
    public var query = ""
    public var submittedQuery = ""
    public var phase: ImageSearchPhase = .idle
    public var ready = false
    public var photos: [SearchPhoto] = []
    public var matches: [ImageSearchMatch] = []
    public var hasSearched = false
    public var showsAllCandidates = false
    public var completed = 0
    public var total = 0
    public var seconds: Double?
    public var failure: ImageSearchFailure?
    public var session = 0
    public var selection: SearchPhoto?

    public init(modelFolder: URL? = nil) { self.modelFolder = modelFolder }
    public var isBusy: Bool { phase != .idle }
    public var displayedMatches: [ImageSearchMatch] {
        showsAllCandidates ? matches : matches.filter { $0.similarity >= Self.minimumSimilarity }
    }
    public var canSearch: Bool {
        ready && !photos.isEmpty && !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isBusy
    }
}

public enum ImageSearchAction: Sendable {
    case setQuery(String)
    case load(URL)
    case loaded(Int)
    case index([URL])
    case progress(Int, Int)
    case indexed(Int, [SearchPhoto], Double)
    case search
    case searched(Int, [ImageSearchMatch], Double)
    case stop
    case stopped(Int)
    case disappeared
    case failed(Int, ImageSearchFailure)
    case selectionFailed(String)
    case select(SearchPhoto?)
    case showAllCandidates(Bool)
}

public enum ImageSearchEffect: Equatable, Sendable {
    case load(Int, URL)
    case index(Int, [URL])
    case search(Int, String)
    case waitForIdle(Int)
    case close(Int)
}

public enum ImageSearchEffectID: Hashable, Sendable { case work, control }

public enum ImageSearchFeature: Feature {
    public typealias State = ImageSearchState
    public typealias Action = ImageSearchAction
    public typealias Effect = ImageSearchEffect
    public typealias EffectID = ImageSearchEffectID

    public static func reduce(_ state: inout State, _ action: Action) -> [Command<Effect, EffectID>] {
        switch action {
        case .setQuery(let value): state.query = value
        case .load(let folder):
            guard !state.isBusy else { return [] }
            state.modelFolder = folder
            state.ready = false
            state.photos = []
            state.matches = []
            state.hasSearched = false
            state.showsAllCandidates = false
            state.selection = nil
            start(&state, phase: .loading)
            return [.run(.load(state.session, folder), id: .work)]
        case .loaded(let id):
            guard id == state.session, state.phase == .loading else { return [] }
            state.ready = true
            state.phase = .idle
        case .index(let urls):
            guard state.ready, !state.isBusy, !urls.isEmpty else { return [] }
            guard urls.count <= 50 else {
                state.failure = ImageSearchFailure(message: "Choose up to 50 images at a time.")
                return []
            }
            var seen = Set<URL>()
            let unique = urls.filter { seen.insert($0).inserted }
            state.photos = []
            state.matches = []
            state.hasSearched = false
            state.showsAllCandidates = false
            state.selection = nil
            state.completed = 0
            state.total = unique.count
            start(&state, phase: .indexing)
            return [.run(.index(state.session, unique), id: .work)]
        case .progress(let id, let completed):
            guard id == state.session, state.phase == .indexing else { return [] }
            state.completed = min(state.total, max(0, completed))
        case .indexed(let id, let photos, let seconds):
            guard id == state.session, state.phase == .indexing else { return [] }
            state.photos = photos
            state.seconds = seconds
            state.phase = .idle
        case .search:
            guard state.canSearch else { return [] }
            state.submittedQuery = state.query
            state.matches = []
            state.hasSearched = false
            state.showsAllCandidates = false
            start(&state, phase: .searching)
            return [.run(.search(state.session, state.submittedQuery), id: .work)]
        case .searched(let id, let matches, let seconds):
            guard id == state.session, state.phase == .searching else { return [] }
            state.matches = matches
            state.hasSearched = true
            state.seconds = seconds
            state.phase = .idle
        case .stop:
            guard state.isBusy, state.phase != .stopping else { return [] }
            state.session += 1
            state.phase = .stopping
            return [.cancel(.work), .run(.waitForIdle(state.session), id: .control)]
        case .stopped(let id):
            guard id == state.session, state.phase == .stopping else { return [] }
            state.phase = .idle
        case .disappeared:
            state.session += 1
            state.ready = false
            state.photos = []
            state.matches = []
            state.hasSearched = false
            state.showsAllCandidates = false
            state.selection = nil
            state.phase = .stopping
            return [.cancelAll, .run(.close(state.session), id: .control)]
        case .failed(let id, let failure):
            guard id == state.session, state.isBusy else { return [] }
            state.failure = failure
            state.phase = .idle
        case .selectionFailed(let detail):
            guard !state.isBusy else { return [] }
            state.failure = ImageSearchFailure(message: "The selected files could not be opened.", detail: detail)
        case .select(let photo): state.selection = photo
        case .showAllCandidates(let value): state.showsAllCandidates = value
        }
        return []
    }

    private static func start(_ state: inout State, phase: ImageSearchPhase) {
        state.session += 1
        state.phase = phase
        state.failure = nil
        state.seconds = nil
    }
}
