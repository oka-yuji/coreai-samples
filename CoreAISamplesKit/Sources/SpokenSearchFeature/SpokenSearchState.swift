public struct SpokenSearchState: Equatable, Sendable {
    public internal(set) var isActive = false
    public internal(set) var filename: String?
    public internal(set) var query = ""
    public internal(set) var hasTranscript = false
    public internal(set) var hits: [SpokenHit] = []
    public internal(set) var hasSearched = false
    public internal(set) var notice: SpokenNotice?
    public internal(set) var generation = 0
    public internal(set) var playingID: Int?
    var isTranscribing = false
    var isSearching = false

    public init() {}

    public var isWorking: Bool { isTranscribing || isSearching }
}
