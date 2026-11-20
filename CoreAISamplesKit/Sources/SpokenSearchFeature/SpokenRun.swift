public struct SpokenRun: Equatable, Sendable {
    public let text: String
    public let start: Double?
    public let end: Double?

    public init(text: String, start: Double?, end: Double?) {
        self.text = text
        self.start = start
        self.end = end
    }
}
