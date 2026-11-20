import Foundation

public enum SpokenSearchEffect: Equatable, Sendable {
    case open(Int, URL)
    case find(Int, String)
    case play(Int, SpokenHit)
    case stopPlayback(Int)
    case clear(Int)
}
