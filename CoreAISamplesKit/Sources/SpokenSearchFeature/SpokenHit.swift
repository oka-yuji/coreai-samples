public struct SpokenHit: Identifiable, Equatable, Sendable {
    public let id: Int
    public let text: String
    public let start: Double
    public let end: Double

    public init(id: Int, text: String, start: Double, end: Double) {
        self.id = id
        self.text = text
        self.start = start
        self.end = end
    }
}
