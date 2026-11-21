import Foundation
import SpeechCore

public enum SpeechStatusMessage: Sendable, Equatable {
    case text(String)
    case tokenLimit(count: Int, limit: Int)

    public var renderedText: String? {
        switch self {
        case .text(let message):
            message
        case .tokenLimit(let count, let limit):
            SpeechError.textExceedsLimit(count: count, limit: limit).errorDescription
        }
    }
}
