import Foundation

public enum SpokenSearchAction: Sendable {
    case appeared
    case disappeared
    case fileSelected(URL, generation: Int)
    case fileSelectionFailed(generation: Int)
    case queryChanged(String)
    case search
    case play(Int)
    case stopPlayback
    case cancel
    case recognized(Int)
    case found(Int, [SpokenHit])
    case playbackEnded(Int, Int)
    case failed(Int, SpokenNotice)
}
