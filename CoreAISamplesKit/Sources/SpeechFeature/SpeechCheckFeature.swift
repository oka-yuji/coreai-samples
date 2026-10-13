import FeatureStore
import Foundation

public struct SpeechCheckRequest: Equatable, Sendable {
    public let folder: String
    public let runID: String
    public init(folder: String, runID: String) {
        self.folder = folder
        self.runID = runID
    }
}
public struct SpeechCheckState: Equatable, Sendable {
    public enum Phase: Sendable, Equatable { case idle, running, stopping, finished }
    public var phase: Phase = .idle
    public var started = false
    public var session = 0
    public var message = "Ready"
    public init() {}
}
public enum SpeechCheckAction: Equatable, Sendable {
    case start(SpeechCheckRequest)
    case progress(Int, String)
    case completed(Int, String)
    case failed(Int, String)
    case stop
    case stopped(Int)
}
public enum SpeechCheckEffect: Equatable, Sendable {
    case run(Int, SpeechCheckRequest)
    case stop(Int)
}
public enum SpeechCheckEffectID: Hashable, Sendable { case run, stop }
public enum SpeechCheckFeature: Feature {
    public typealias State = SpeechCheckState
    public typealias Action = SpeechCheckAction
    public typealias Effect = SpeechCheckEffect
    public typealias EffectID = SpeechCheckEffectID
    public typealias Command = FeatureStore.Command<Effect, EffectID>
    public static func reduce(_ state: inout State, _ action: Action) -> [Command] {
        switch action {
        case .start(let request):
            guard !state.started else { return [] }
            state.started = true
            state.session += 1
            state.phase = .running
            state.message = "Checking the device…"
            return [.run(.run(state.session, request), id: .run)]
        case .progress(let session, let message):
            guard session == state.session, state.phase == .running else { return [] }
            state.message = message
        case .completed(let session, let path):
            guard session == state.session, state.phase == .running else { return [] }
            state.phase = .finished
            state.message = "Device check finished. Results: " + path
        case .failed(let session, let message):
            guard session == state.session, state.phase == .running else { return [] }
            state.phase = .finished
            state.message = "Device check failed: " + message
        case .stop:
            guard state.phase == .running else { return [] }
            state.session += 1
            state.phase = .stopping
            state.message = "Stopping and releasing resources…"
            return [.cancelAll, .run(.stop(state.session), id: .stop)]
        case .stopped(let session):
            guard session == state.session, state.phase == .stopping else { return [] }
            state.phase = .finished
            state.message = "Device check stopped."
        }
        return []
    }
}
