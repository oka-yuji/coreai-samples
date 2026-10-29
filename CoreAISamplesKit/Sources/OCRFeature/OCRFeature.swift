import FeatureStore
import Foundation

public enum OCRExportFormat: String, Equatable, Sendable { case markdown, json }

public struct OCRExport: Equatable, Identifiable, Sendable {
    public let id: Int
    public let session: Int
    public let format: OCRExportFormat
    public let name: String
    public let data: Data
}

public struct OCRState: Equatable, Sendable {
    public enum Phase: Equatable, Sendable {
        case idle, selecting, loading, reading, complete, cancelled, failed, stopping
    }
    public var phase: Phase = .idle
    public var presented = false
    public var session = 0
    public let bundle: URL
    public var input: OCRInput?
    public var document: OCRDocument?
    public var message: String?
    public var export: OCRExport?
    public var exportDismissed = false
    var nextExport = 0

    public init(bundle: URL, input: OCRInput? = nil) {
        self.bundle = bundle
        self.input = input
    }

    public var busy: Bool { [.selecting, .loading, .reading, .stopping].contains(phase) }
    public var exportVisible: Bool { export != nil && !exportDismissed }
    public var editable: Bool { presented && !busy && !exportVisible }
    public var canRead: Bool { editable && input != nil }
}

public enum OCRAction: Equatable, Sendable {
    case appeared, ended
    case selectImage(URL)
    case selected(Int, OCRInput)
    case read
    case loaded(Int)
    case completed(Int, OCRDocument)
    case failed(Int, String)
    case cancelled(Int)
    case stop
    case stopped(Int)
    case exportRequested(OCRExportFormat)
    case exportDismissed(id: Int, session: Int)
    case exportFinished(id: Int, session: Int, error: String?)
    case uiError(String)
}

public enum OCREffect: Equatable, Sendable {
    case select(Int, URL)
    case load(Int, URL)
    case read(Int, OCRInput)
    case release(Int)
}

public enum OCREffectID: Hashable, Sendable { case operation, stop }

public enum OCRFeature: Feature {
    public typealias State = OCRState
    public typealias Action = OCRAction
    public typealias Effect = OCREffect
    public typealias EffectID = OCREffectID
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
            return stop(&state)
        case .selectImage(let url):
            guard state.editable, url.isFileURL else { return [] }
            state.session += 1
            state.phase = .selecting
            state.input = nil
            state.document = nil
            state.message = nil
            clearExport(&state)
            return [.run(.select(state.session, url), id: .operation)]
        case .selected(let session, let input):
            guard state.session == session, state.phase == .selecting else { return [] }
            state.input = input
            state.phase = .idle
        case .read:
            guard state.canRead else { return [] }
            state.session += 1
            state.phase = .loading
            state.document = nil
            state.message = nil
            clearExport(&state)
            return [.run(.load(state.session, state.bundle), id: .operation)]
        case .loaded(let session):
            guard state.session == session, state.phase == .loading, let input = state.input else { return [] }
            state.phase = .reading
            return [.run(.read(session, input), id: .operation)]
        case .completed(let session, let document):
            guard state.session == session, state.phase == .reading else { return [] }
            state.document = document
            state.phase = .complete
        case .failed(let session, let message):
            guard state.session == session, [.selecting, .loading, .reading].contains(state.phase) else { return [] }
            state.phase = .failed
            state.message = message
        case .cancelled(let session):
            guard state.session == session, [.selecting, .loading, .reading].contains(state.phase) else { return [] }
            return stop(&state)
        case .stop:
            guard state.phase != .stopping else { return [] }
            return stop(&state)
        case .stopped(let session):
            guard state.session == session, state.phase == .stopping else { return [] }
            state.phase = state.presented ? .cancelled : .idle
        case .exportRequested(let format):
            guard state.editable, let document = state.document else { return [] }
            clearExport(&state)
            state.message = nil
            do {
                let data = try format == .json ? document.json() : Data((document.markdown + "\n").utf8)
                state.nextExport += 1
                state.export = OCRExport(
                    id: state.nextExport, session: state.session, format: format,
                    name: URL(filePath: document.image).deletingPathExtension().lastPathComponent, data: data)
                state.exportDismissed = false
            } catch { state.message = "The OCR result could not be prepared for export." }
        case .exportDismissed(let id, let session):
            guard state.export?.id == id, state.export?.session == session else { return [] }
            state.exportDismissed = true
        case .exportFinished(let id, let session, let error):
            guard state.export?.id == id, state.export?.session == session else { return [] }
            clearExport(&state)
            state.message = error
        case .uiError(let message):
            guard !state.busy else { return [] }
            state.message = message
        }
        return []
    }

    private static func stop(_ state: inout State) -> [Command] {
        state.session += 1
        state.phase = .stopping
        state.message = nil
        clearExport(&state)
        return [.cancelAll, .run(.release(state.session), id: .stop)]
    }

    private static func clearExport(_ state: inout State) {
        state.export = nil
        state.exportDismissed = false
    }
}
