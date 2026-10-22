import FeatureStore
import Foundation

public struct VideoMoment: Equatable, Sendable, Identifiable {
    public let time: Double
    public let thumbnail: Data
    public var id: Double { time }
    public init(time: Double, thumbnail: Data) {
        self.time = time
        self.thumbnail = thumbnail
    }
}

public struct VideoSearchMatch: Equatable, Sendable, Identifiable {
    public let moment: VideoMoment
    public let similarity: Double
    public var id: Double { moment.id }
    public init(moment: VideoMoment, similarity: Double) {
        self.moment = moment
        self.similarity = similarity
    }
}

public struct VideoMomentSelection: Equatable, Sendable {
    public let index: Int
    public let previous: VideoSearchMatch?
    public let next: VideoSearchMatch?
}

public struct VideoIndex: Equatable, Sendable {
    public let duration: Double
    public let moments: [VideoMoment]
    public init(duration: Double, moments: [VideoMoment]) {
        self.duration = duration
        self.moments = moments
    }
}

public struct VideoSearchFailure: Equatable, Sendable {
    public let message: String
    public let detail: String
    public init(message: String, detail: String = "") {
        self.message = message
        self.detail = detail
    }
}

public enum VideoSearchPhase: Equatable, Sendable {
    case idle, loading, indexing, searching, stopping
}

public enum VideoSearchExecution: Equatable, Sendable {
    case standalone, managed
}

public enum VideoSampling {
    public static func times(duration: Double, interval: Double = 1, limit: Int = 300) -> [Double] {
        guard duration.isFinite, duration > 0, interval.isFinite, interval > 0, limit > 0 else { return [] }
        let intervals = max(1, (duration / interval).rounded(.up))
        guard intervals <= Double(limit) else {
            let step = duration / Double(limit)
            return (0..<limit).map { (Double($0) + 0.5) * step }
        }
        return (0..<Int(intervals)).map { index in
            let start = Double(index) * interval
            return (start + min(duration, start + interval)) / 2
        }
    }
}

public struct VideoSearchState: Equatable, Sendable {
    public static let minimumGap = 3.0
    public static let thresholdRange = 0.3...0.9
    public var execution: VideoSearchExecution = .standalone
    public var modelFolder: URL?
    public var resourceIdentity: String?
    public var indexIdentity: String?
    public var resultIdentity: String?
    public var video: URL?
    public var duration = 0.0
    public var moments: [VideoMoment] = []
    public var matches: [VideoSearchMatch] = []
    public var query = ""
    public var submittedQuery = ""
    public var phase: VideoSearchPhase = .idle
    public var ready = false
    public var hasSearched = false
    public var showsAllCandidates = false
    public var minimumSimilarity = 0.67
    public var completed = 0
    public var total = 0
    public var seconds: Double?
    public var failure: VideoSearchFailure?
    public var session = 0
    public internal(set) var pendingIndex: URL?

    public init(modelFolder: URL? = nil) { self.modelFolder = modelFolder }

    public init(managed folder: URL, identity: String, retaining previous: VideoSearchState? = nil) {
        execution = .managed
        modelFolder = folder
        resourceIdentity = identity
        guard let previous else { return }
        session = previous.session
        video = previous.video
        duration = previous.duration
        moments = previous.moments
        matches = previous.matches
        query = previous.query
        submittedQuery = previous.submittedQuery
        hasSearched = previous.hasSearched
        showsAllCandidates = previous.showsAllCandidates
        resultIdentity = previous.resultIdentity
        minimumSimilarity = previous.minimumSimilarity
    }

    public var isBusy: Bool { phase != .idle }
    public var currentIndexIdentity: String {
        [resourceIdentity ?? "", video?.absoluteString ?? ""].joined(separator: "\n")
    }
    public var hasValidIndex: Bool { ready && !moments.isEmpty && indexIdentity == currentIndexIdentity }
    public var needsIndex: Bool { video != nil && !hasValidIndex }
    public var canIndex: Bool { !isBusy && modelFolder != nil }
    public var canChangeSettings: Bool { !isBusy }
    public var canSearch: Bool {
        hasValidIndex && !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isBusy
    }
    public var displayedMatches: [VideoSearchMatch] {
        if showsAllCandidates { return matches }
        return Self.suppressNeighbors(matches.filter { $0.similarity >= minimumSimilarity }, gap: Self.minimumGap)
    }

    public static func suppressNeighbors(_ matches: [VideoSearchMatch], gap: Double) -> [VideoSearchMatch] {
        let ranked = matches.sorted {
            $0.similarity == $1.similarity ? $0.moment.time < $1.moment.time : $0.similarity > $1.similarity
        }
        var kept: [VideoSearchMatch] = []
        for match in ranked where !kept.contains(where: { abs($0.moment.time - match.moment.time) <= gap }) {
            kept.append(match)
        }
        return kept
    }

    public static func selection(in matches: [VideoSearchMatch], at time: Double?) -> VideoMomentSelection? {
        guard let time, let index = matches.firstIndex(where: { $0.moment.time == time }) else { return nil }
        return VideoMomentSelection(
            index: index,
            previous: index > 0 ? matches[index - 1] : nil,
            next: index + 1 < matches.count ? matches[index + 1] : nil)
    }
}

public enum VideoSearchAction: Sendable {
    case setQuery(String)
    case load(URL)
    case loaded(Int)
    case index(URL)
    case videoChosen(Int, URL)
    case progress(Int, Int, Int)
    case indexed(Int, [VideoMoment], Double, Double)
    case search
    case searched(Int, [VideoSearchMatch], Double)
    case stop
    case stopped(Int)
    case disappeared
    case closed
    case failed(Int, VideoSearchFailure)
    case closeFailed(Int, String)
    case selectionFailed(Int, String)
    case showAllCandidates(Bool)
    case thresholdChanged(Double)
}

public enum VideoSearchEffect: Equatable, Sendable {
    case load(Int, URL)
    case index(Int, URL)
    case search(Int, String)
    case waitForIdle(Int)
    case close(Int)
}

public enum VideoSearchEffectID: Hashable, Sendable { case work, control }

public enum VideoSearchFeature: Feature {
    public typealias State = VideoSearchState
    public typealias Action = VideoSearchAction
    public typealias Effect = VideoSearchEffect
    public typealias EffectID = VideoSearchEffectID

    public static func reduce(_ state: inout State, _ action: Action) -> [Command<Effect, EffectID>] {
        switch action {
        case .setQuery(let value): state.query = value
        case .load(let folder):
            guard state.execution == .standalone, !state.isBusy else { return [] }
            if folder != state.modelFolder {
                discardVideo(&state)
                state.resultIdentity = nil
            }
            state.indexIdentity = nil
            state.modelFolder = folder
            state.ready = false
            start(&state, phase: .loading)
            return [.run(.load(state.session, folder), id: .work)]
        case .loaded(let id):
            guard id == state.session, state.phase == .loading else { return [] }
            state.ready = true
            guard let url = state.pendingIndex else {
                state.phase = .idle
                return []
            }
            state.pendingIndex = nil
            state.phase = .indexing
            return [.run(.index(id, url), id: .work)]
        case .index(let url):
            guard state.canIndex else { return [] }
            if url != state.video {
                discardVideo(&state)
                state.video = url
            }
            discardResults(&state)
            state.completed = 0
            state.total = 0
            let folder = state.ready ? nil : state.modelFolder
            start(&state, phase: folder == nil ? .indexing : .loading)
            guard let folder else { return [.run(.index(state.session, url), id: .work)] }
            state.pendingIndex = url
            return [.run(.load(state.session, folder), id: .work)]
        case .videoChosen(let request, let url):
            guard request == state.session else { return [] }
            return reduce(&state, .index(url))
        case .progress(let id, let completed, let total):
            guard id == state.session, state.phase == .indexing else { return [] }
            state.total = max(0, total)
            state.completed = min(state.total, max(0, completed))
        case .indexed(let id, let moments, let duration, let seconds):
            guard id == state.session, state.phase == .indexing else { return [] }
            state.moments = moments
            state.duration = duration
            state.indexIdentity = state.currentIndexIdentity
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
            state.resultIdentity = state.resourceIdentity
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
        case .disappeared, .closed:
            state.session += 1
            state.ready = false
            state.indexIdentity = nil
            state.failure = nil
            state.phase = .stopping
            return [.cancelAll, .run(.close(state.session), id: .control)]
        case .failed(let id, let failure):
            guard id == state.session, state.isBusy else { return [] }
            state.failure = failure
            state.phase = .idle
        case .closeFailed(let id, let detail):
            guard id == state.session, state.phase == .stopping else { return [] }
            state.failure = VideoSearchFailure(message: "The model could not be released.", detail: detail)
            state.phase = .idle
        case .selectionFailed(let request, let detail):
            guard request == state.session, !state.isBusy else { return [] }
            state.failure = VideoSearchFailure(message: "The selected video could not be opened.", detail: detail)
        case .showAllCandidates(let value): state.showsAllCandidates = value
        case .thresholdChanged(let value):
            guard state.canChangeSettings, VideoSearchState.thresholdRange.contains(value) else { return [] }
            state.minimumSimilarity = value
        }
        return []
    }

    private static func discardVideo(_ state: inout State) {
        state.video = nil
        state.duration = 0
        state.moments = []
        discardResults(&state)
    }

    private static func discardResults(_ state: inout State) {
        state.matches = []
        state.hasSearched = false
        state.showsAllCandidates = false
        state.indexIdentity = nil
    }

    private static func start(_ state: inout State, phase: VideoSearchPhase) {
        state.session += 1
        state.phase = phase
        state.failure = nil
        state.seconds = nil
        state.pendingIndex = nil
    }
}
