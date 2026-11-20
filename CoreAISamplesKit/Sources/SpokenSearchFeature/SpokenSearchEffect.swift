import Foundation

public enum SpokenSearchEffect: Equatable, Sendable {
    case open(Int, URL)
    case find(Int, String)
    case findTopic(Int, String, String)
    case waitForTopic(Int)
    case play(Int, SpokenHit, topicIdentity: String? = nil)
    case stopPlayback(Int)
    case clear(Int)
}
