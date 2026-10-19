import Foundation
import TruckInspectionFeature

public enum InspectionSpeechEvent: Sendable {
    case partial(String, elapsed: Double)
    case final(String)
    case level(Double, elapsed: Double)
    case route(String)
    case tick(Double)
    case failure(String)
}

public enum InspectionSpeechError: Error {
    case unavailable, microphoneDenied, speechRecognitionDenied, noJapaneseVoice, routeUnavailable, invalidAudioFormat
    case playbackTimedOut
    case recognition(String)

    var limit: InspectionVoiceLimit? {
        switch self {
        case .microphoneDenied: .microphoneDenied
        case .speechRecognitionDenied: .speechRecognitionDenied
        case .unavailable: .recognitionUnavailable
        case .noJapaneseVoice: .voiceUnavailable
        case .routeUnavailable, .invalidAudioFormat, .playbackTimedOut, .recognition: nil
        }
    }
}

@MainActor
public protocol TruckInspectionSpeechService {
    func prepare() async throws -> String
    func speak(_ text: String) async throws
    func startListening(hints: [String]) async throws -> AsyncStream<InspectionSpeechEvent>
    func endAudio() async
    func finishListening() async
    func stop() async throws
}
