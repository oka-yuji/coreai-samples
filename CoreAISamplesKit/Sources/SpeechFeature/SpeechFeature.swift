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
    public enum Phase: Sendable, Equatable {
        case idle, opening, generating, playing, stopping
        case preparingRecording, recording, finishingRecording, importingReference
    }
    public var phase: Phase = .idle
    public var presented = false
    public var managed = false
    public var session = 0
    public var text: String
    public var seed = "1708"
    public var model: SpeechModel?
    public var audio: SpeechAudio?
    public var reference: SpeechReferenceAudio?
    public var referenceMessage: String?
    public var microphoneDenied = false
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
            && (reference == nil || model?.supportsReferenceAudio == true)
    }
    public var canPlay: Bool { editable && audio != nil }
    public var canSelectReference: Bool { editable && model?.supportsReferenceAudio == true }
    public var canPreviewReference: Bool { editable && reference != nil }
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
    case recordReference, finishReferenceRecording, clearReference, playReference
    case importReference(Int, URL)
    case referenceRecordingStarted(Int)
    case referencePrepared(Int, SpeechReferenceAudio)
    case microphoneDenied(Int)
    case referenceSelectionFailed(Int, String)
}

public enum SpeechEffect: Equatable, Sendable {
    case open(Int, String)
    case generate(Int, String, UInt64)
    case play(Int, Data)
    case stop(Int)
    case generateReference(Int, String, UInt64, SpeechReferenceAudio)
    case recordReference(Int)
    case finishReferenceRecording(Int)
    case importReference(Int, URL)
    case playReference(Int, SpeechReferenceAudio)
}

public enum SpeechEffectID: Hashable, Sendable { case operation, stop, recordingControl }

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
                state.reference = nil
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
            state.reference = nil
            state.referenceMessage = nil
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
            if let reference = state.reference {
                return [.run(.generateReference(state.session, state.text, seed, reference), id: .operation)]
            }
            return [.run(.generate(state.session, state.text, seed), id: .operation)]
        case .generated(let session, let audio):
            guard state.session == session, state.phase == .generating else { return [] }
            state.phase = .idle
            state.audio = audio
            state.message = "Speech is ready. Play it or save the WAV file."
        case .failed(let session, let message):
            guard state.session == session,
                [
                    .opening, .generating, .playing, .preparingRecording, .recording, .finishingRecording,
                    .importingReference,
                ]
                .contains(state.phase)
            else { return [] }
            if [.preparingRecording, .recording, .finishingRecording, .importingReference].contains(state.phase) {
                state.referenceMessage = message
                state.message = nil
            } else {
                state.message = message
            }
            state.phase = .idle
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
            let finishingRecording = state.phase == .finishingRecording
            state.session += 1
            state.phase = .stopping
            var commands: [Command] = [.cancel(.operation)]
            if finishingRecording { commands.append(.cancel(.recordingControl)) }
            commands.append(.run(.stop(state.session), id: .stop))
            return commands
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
        case .recordReference:
            guard state.canSelectReference else { return [] }
            invalidateAudio(&state)
            state.microphoneDenied = false
            state.referenceMessage = nil
            state.phase = .preparingRecording
            return [.run(.recordReference(state.session), id: .operation)]
        case .referenceSelectionFailed(let selectionSession, let message):
            guard state.canSelectReference, selectionSession == state.session else { return [] }
            state.referenceMessage = message
        case .referenceRecordingStarted(let session):
            guard state.session == session, state.phase == .preparingRecording else { return [] }
            state.phase = .recording
        case .finishReferenceRecording:
            guard state.phase == .recording else { return [] }
            state.phase = .finishingRecording
            return [.run(.finishReferenceRecording(state.session), id: .recordingControl)]
        case .importReference(let selectionSession, let url):
            guard state.canSelectReference, selectionSession == state.session else { return [] }
            invalidateAudio(&state)
            state.microphoneDenied = false
            state.referenceMessage = nil
            state.phase = .importingReference
            return [.run(.importReference(state.session, url), id: .operation)]
        case .referencePrepared(let session, let reference):
            guard state.session == session,
                [.preparingRecording, .recording, .finishingRecording, .importingReference].contains(state.phase)
            else { return [] }
            state.phase = .idle
            state.reference = reference
            state.referenceMessage = nil
            state.message = "Reference voice is ready. Enter text and generate speech."
        case .microphoneDenied(let session):
            guard state.session == session, state.phase == .preparingRecording else { return [] }
            state.phase = .idle
            state.microphoneDenied = true
            state.referenceMessage = SpeechReferenceError.permissionDenied.errorDescription
        case .clearReference:
            guard state.editable else { return [] }
            state.reference = nil
            state.referenceMessage = nil
            invalidateAudio(&state)
        case .playReference:
            guard state.canPreviewReference, let reference = state.reference else { return [] }
            state.session += 1
            state.phase = .playing
            state.message = nil
            return [.run(.playReference(state.session, reference), id: .operation)]
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
