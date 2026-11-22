struct PreparedCopyMark: Codable, Equatable, Sendable {
    enum Kind: String, Codable, Sendable {
        case normal, recovery
    }

    enum Stage: String, Codable, Sendable {
        case load, warmUp
    }

    var attempt: String
    var pid: Int32
    var start: Int64
    var kind: Kind
    var stage: Stage
    var at: Double
}
