import FeatureStore
import Foundation
import SpeechCore

public struct SpeechExample: Codable, Identifiable, Sendable, Equatable {
    public let id: String
    public let title: String
    public let text: String
    public init(id: String, title: String, text: String) {
        self.id = id
        self.title = title
        self.text = text
    }
}

public struct SpeechExportTicket: Identifiable, Sendable, Equatable {
    public let id: Int
    public let session: Int
    public init(id: Int, session: Int) {
        self.id = id
        self.session = session
    }
}

public struct SpeechState: Sendable, Equatable {
    public enum Phase: Sendable, Equatable { case idle, opening, generating, playing, stopping }
    public var phase: Phase = .idle
    public var presented = false
    public var managed = false
    public var session = 0
    public var text: String
    public var seed = "1708"
    public var model: SpeechModel?
    public var audio: SpeechAudio?
    public var message: String?
    public var examples: [SpeechExample]
    public var lastPath: String?
    public var exportTicket: SpeechExportTicket?
    public var exportDismissed = false
    public var nextExport = 0

    public init(examples: [SpeechExample] = [], lastPath: String? = nil) {
        self.examples = examples
        self.text = examples.first?.text ?? ""
        self.lastPath = lastPath
    }
    public init(managedModel: SpeechModel, examples: [SpeechExample] = [], retaining previous: SpeechState? = nil) {
        self.init(examples: examples)
        managed = true
        model = managedModel
        text = previous?.text ?? text
        seed = previous?.seed ?? seed
    }
    public var busy: Bool { phase != .idle }
    public var exportVisible: Bool { exportTicket != nil && !exportDismissed }
    public var editable: Bool { presented && !busy && !exportVisible }
    public var canGenerate: Bool {
        editable && model != nil && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && UInt64(seed) != nil
    }
    public var canPlay: Bool { editable && audio != nil }
}

public enum SpeechAction: Equatable, Sendable {
    case appeared, ended
    case openBundle(String)
    case opened(Int, SpeechModel)
    case setText(String)
    case setSeed(String)
    case generate
    case generated(Int, SpeechAudio)
    case failed(Int, String)
    case play, stop
    case playbackFinished(Int)
    case playbackInterrupted(Int)
    case stopped(Int)
    case prepareExport
    case exportDismissed(SpeechExportTicket)
    case exportFinished(SpeechExportTicket, String?)
    case uiError(String)
}

public enum SpeechEffect: Equatable, Sendable {
    case open(Int, String)
    case generate(Int, String, UInt64)
    case play(Int, Data)
    case stop(Int)
}

public enum SpeechEffectID: Hashable, Sendable { case operation, stop }

public enum SpeechFeature: Feature {
    public typealias State = SpeechState
    public typealias Action = SpeechAction
    public typealias Effect = SpeechEffect
    public typealias EffectID = SpeechEffectID
    public typealias Command = FeatureStore.Command<Effect, EffectID>

    public static func reduce(_ state: inout State, _ action: Action) -> [Command] {
        if !state.presented {
            switch action {
            case .appeared, .stopped: break
            default: return []
            }
        }
        switch action {
        case .appeared:
            state.presented = true
        case .ended:
            state.presented = false
            if state.managed {
                state.audio = nil
                state.message = nil
            }
            state.session += 1
            state.phase = .stopping
            clearExport(&state)
            return [.cancelAll, .run(.stop(state.session), id: .stop)]
        case .openBundle(let path):
            guard !state.managed, state.editable else { return [] }
            state.session += 1
            state.phase = .opening
            state.model = nil
            state.audio = nil
            state.message = nil
            clearExport(&state)
            return [.run(.open(state.session, path), id: .operation)]
        case .opened(let session, let model):
            guard state.session == session, state.phase == .opening else { return [] }
            state.phase = .idle
            state.model = model
            state.lastPath = model.directory.path
            state.message = "Model files verified. Enter text and generate speech."
        case .setText(let text):
            guard state.editable else { return [] }
            state.text = text
            invalidateAudio(&state)
        case .setSeed(let seed):
            guard state.editable else { return [] }
            state.seed = seed
            invalidateAudio(&state)
        case .generate:
            guard state.canGenerate, let seed = UInt64(state.seed) else { return [] }
            state.session += 1
            state.phase = .generating
            state.audio = nil
            state.message = nil
            clearExport(&state)
            return [.run(.generate(state.session, state.text, seed), id: .operation)]
        case .generated(let session, let audio):
            guard state.session == session, state.phase == .generating else { return [] }
            state.phase = .idle
            state.audio = audio
            state.message = "Speech is ready. Play it or save the WAV file."
        case .failed(let session, let message):
            guard state.session == session, [.opening, .generating, .playing].contains(state.phase) else { return [] }
            state.phase = .idle
            state.message = message
        case .play:
            guard state.canPlay, let audio = state.audio else { return [] }
            state.session += 1
            state.phase = .playing
            state.message = nil
            clearExport(&state)
            return [.run(.play(state.session, audio.wav), id: .operation)]
        case .playbackFinished(let session):
            guard state.session == session, state.phase == .playing else { return [] }
            state.phase = .idle
            state.message = "Playback finished."
        case .playbackInterrupted(let session):
            guard state.session == session, state.phase == .playing else { return [] }
            state.phase = .idle
            state.message = "Playback interrupted. You can play the audio again."
        case .stop:
            guard state.busy, state.phase != .stopping else { return [] }
            state.session += 1
            state.phase = .stopping
            return [.cancel(.operation), .run(.stop(state.session), id: .stop)]
        case .stopped(let session):
            guard state.session == session, state.phase == .stopping else { return [] }
            state.phase = .idle
            state.message = "Stopped."
        case .prepareExport:
            guard state.canPlay else { return [] }
            state.nextExport += 1
            state.exportTicket = SpeechExportTicket(id: state.nextExport, session: state.session)
            state.exportDismissed = false
        case .exportDismissed(let ticket):
            guard ticket == state.exportTicket, ticket.session == state.session else { return [] }
            state.exportDismissed = true
        case .exportFinished(let ticket, let error):
            guard ticket == state.exportTicket, ticket.session == state.session else { return [] }
            clearExport(&state)
            state.message = error ?? "WAV file saved."
        case .uiError(let message):
            guard !state.busy else { return [] }
            state.message = message
        }
        return []
    }

    private static func invalidateAudio(_ state: inout State) {
        state.session += 1
        state.audio = nil
        state.message = nil
        clearExport(&state)
    }
    private static func clearExport(_ state: inout State) {
        state.exportTicket = nil
        state.exportDismissed = false
    }
}
