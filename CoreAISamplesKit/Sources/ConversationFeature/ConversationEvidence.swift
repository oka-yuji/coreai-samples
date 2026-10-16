import Foundation

public struct ConversationEvidence: Codable, Equatable, Sendable {
    public struct Page: Codable, Equatable, Sendable {
        public var url: String
        public var finalURL: String
        public var sha256: String
        public var characters: Int

        public init(url: String, finalURL: String, sha256: String, characters: Int) {
            self.url = url
            self.finalURL = finalURL
            self.sha256 = sha256
            self.characters = characters
        }
    }

    public struct Attempt: Codable, Equatable, Sendable {
        public var url: String
        public var ok: Bool

        public init(url: String, ok: Bool) {
            self.url = url
            self.ok = ok
        }
    }

    public var pages: [Page]
    public var attempts: [Attempt]
    public var seen: [String: String]

    public init(pages: [Page] = [], attempts: [Attempt] = [], seen: [String: String] = [:]) {
        self.pages = pages
        self.attempts = attempts
        self.seen = seen
    }
}
