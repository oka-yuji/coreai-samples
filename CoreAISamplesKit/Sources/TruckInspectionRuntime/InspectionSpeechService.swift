import Foundation

public enum InspectionSpeechEvent: Sendable {
    case partial(String, elapsed: Double)
    case final(String)
    case level(Double, elapsed: Double)
    case route(String)
    case tick(Double)
    case failure(String)
}

public enum InspectionSpeechError: Error {
    case unavailable, permissionDenied, noJapaneseVoice, routeUnavailable, invalidAudioFormat, playbackTimedOut
    case recognition(String)
}

@MainActor
public protocol TruckInspectionSpeechService {
    func prepare() async throws -> String
    func speak(_ text: String) async throws
    func startListening(hints: [String]) async throws -> AsyncStream<InspectionSpeechEvent>
    func endAudio() async
    func finishListening() async
    func stop() async -> String?
}
