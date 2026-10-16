import FeatureStore
import Foundation

public struct SpeechFolder: Equatable, Sendable, Identifiable {
    public let url: URL
    public var id: String { url.path }
    public var name: String { url.lastPathComponent }
    public init(url: URL) { self.url = url }
}

public struct SpeechLibraryState: Equatable, Sendable {
    public var presented = false
    public var loading = false
    public var session = 0
    public var folders: [SpeechFolder] = []
    public var message: String?
    public init() {}
}

public enum SpeechLibraryAction: Equatable, Sendable {
    case appeared, ended, refresh
    case loaded(Int, [SpeechFolder])
    case failed(Int, String)
}
public enum SpeechLibraryEffect: Equatable, Sendable { case load(Int) }
public enum SpeechLibraryEffectID: Hashable, Sendable { case load }

public enum SpeechLibraryFeature: Feature {
    public typealias State = SpeechLibraryState
    public typealias Action = SpeechLibraryAction
    public typealias Effect = SpeechLibraryEffect
    public typealias EffectID = SpeechLibraryEffectID
    public typealias Command = FeatureStore.Command<Effect, EffectID>
    public static func reduce(_ state: inout State, _ action: Action) -> [Command] {
        switch action {
        case .appeared:
            state.presented = true
            return load(&state)
        case .ended:
            state.presented = false
            state.session += 1
            state.loading = false
            return [.cancelAll]
        case .refresh:
            guard state.presented else { return [] }
            return load(&state)
        case .loaded(let session, let folders):
            guard state.presented, state.session == session else { return [] }
            state.loading = false
            state.folders = folders
            state.message = folders.isEmpty ? "No model folders found." : nil
        case .failed(let session, let message):
            guard state.presented, state.session == session else { return [] }
            state.loading = false
            state.message = message
        }
        return []
    }
    private static func load(_ state: inout State) -> [Command] {
        state.session += 1
        state.loading = true
        state.message = nil
        return [.run(.load(state.session), id: .load)]
    }
}
