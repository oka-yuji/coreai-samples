import FeatureStore
import Foundation
import ObjectFinderCore

public enum ObjectFinderPhase: Equatable, Sendable {
    case idle, starting, searching, stopping
}

public enum ObjectFinderFailure: Error, Equatable, Sendable {
    case cameraDenied, cameraUnavailable, cameraInterrupted, notPermitted
    case operation(String)
}

public enum ObjectFinderExecution: Equatable, Sendable {
    case standalone, managed
}

public struct ObjectFinderMemory: Equatable, Sendable {
    public let footprintBytes: UInt64
    public let availableBytes: UInt64

    public init(footprintBytes: UInt64, availableBytes: UInt64) {
        self.footprintBytes = footprintBytes
        self.availableBytes = availableBytes
    }
}

public struct ObjectFinderFrame: Equatable, Sendable {
    public let candidates: [ObjectCandidate]
    public let seconds: Double
    public let rotation: Double
    public let orientationRevision: Int
    public let captureGeneration: Int
    public let similarities: [Double]
    public let capturedAt: Double
    public let memory: ObjectFinderMemory?

    public init(
        candidates: [ObjectCandidate], seconds: Double, rotation: Double = 90, orientationRevision: Int = 0,
        captureGeneration: Int = 0, similarities: [Double] = [], capturedAt: Double = 0,
        memory: ObjectFinderMemory? = nil
    ) {
        self.candidates = candidates
        self.seconds = seconds
        self.rotation = rotation
        self.orientationRevision = orientationRevision
        self.captureGeneration = captureGeneration
        self.similarities = similarities
        self.capturedAt = capturedAt
        self.memory = memory
    }
}

public struct ObjectFinderState: Equatable, Sendable {
    public let execution: ObjectFinderExecution
    public var folder: URL?
    public var query = ""
    public var submittedQuery = ""
    public var minimumSimilarity = 0.67
    public var maximumBoxes = 3
    public var searchMode = ObjectSearchMode.standard
    public var phase: ObjectFinderPhase = .idle
    public var frame: ObjectFinderFrame?
    public var matches: [ObjectCandidate] = []
    public var failure: ObjectFinderFailure?
    public var session = 0
    public var presented = true
    public var stableFrames = 0
    public var emptyFrames = 0
    public var notified = false
    public var orientation = ObjectCameraOrientation()
    public var orientationRevision = 0
    public var restartAfterStop = false
    public var releasePending = false
    public var log: [ObjectFinderFrame] = []
    public var analyzedFrames = 0
    public var peakFootprintBytes: UInt64?

    public static let logLimit = 20
    public static let boxLimits = 1...3

    public var cameraRotation: ObjectCameraRotation {
        ObjectCameraRotation(session: session, revision: orientationRevision, angle: orientation.angle)
    }

    public init(folder: URL? = nil, execution: ObjectFinderExecution = .standalone) {
        self.folder = folder
        self.execution = execution
    }

    public static func managed(folder: URL, retaining previous: Self? = nil) -> Self {
        var state = Self(folder: folder, execution: .managed)
        if let previous {
            state.query = previous.query
            state.submittedQuery = previous.submittedQuery
            state.minimumSimilarity = previous.minimumSimilarity
            state.maximumBoxes = previous.maximumBoxes
            state.searchMode = previous.searchMode
            state.session = previous.session
        }
        return state
    }

    public var canStart: Bool {
        presented && phase == .idle && folder != nil && validDraft && !releasePending
    }

    public var canApplyQuery: Bool {
        presented && (phase == .starting || phase == .searching) && folder != nil && validDraft
            && query.trimmingCharacters(in: .whitespacesAndNewlines) != submittedQuery
    }

    public var showsCamera: Bool {
        phase == .starting || phase == .searching || (phase == .stopping && restartAfterStop)
    }

    public var averageSeconds: Double? {
        log.isEmpty ? nil : log.map(\.seconds).reduce(0, +) / Double(log.count)
    }

    public var framesPerSecond: Double? {
        guard let newest = log.first, let oldest = log.last, newest.capturedAt > oldest.capturedAt else { return nil }
        return Double(log.count - 1) / (newest.capturedAt - oldest.capturedAt)
    }

    public var canStop: Bool {
        phase == .starting || phase == .searching || (phase == .stopping && restartAfterStop)
            || (phase == .idle && releasePending)
    }

    private var validDraft: Bool {
        !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && query.utf8.count <= 512
    }
}

public enum ObjectFinderAction: Sendable {
    case queryChanged(String)
    case folderSelected(URL)
    case thresholdChanged(Double)
    case maximumBoxesChanged(Int)
    case searchModeChanged(ObjectSearchMode)
    case inputFailed(String)
    case poseChanged(ObjectCameraPose)
    case start, applyQuery, stop, disappeared, appeared, backgrounded
    case started(Int)
    case analyzed(Int, ObjectFinderFrame)
    case failed(Int, ObjectFinderFailure)
    case stopFailed(Int, ObjectFinderFailure)
    case stopped(Int)
}

public enum ObjectFinderEffect: Equatable, Sendable {
    case search(Int, URL, String, Double, ObjectSearchMode, ObjectCameraRotation)
    case orient(ObjectCameraRotation)
    case stop(Int, retainingModel: Bool)
    case notify(Int)
}

public enum ObjectFinderEffectID: Hashable, Sendable { case search, control, notification, orientation }

public enum ObjectFinderFeature: Feature {
    public typealias State = ObjectFinderState
    public typealias Action = ObjectFinderAction
    public typealias Effect = ObjectFinderEffect
    public typealias EffectID = ObjectFinderEffectID

    public static func reduce(_ state: inout State, _ action: Action) -> [Command<Effect, EffectID>] {
        switch action {
        case .queryChanged(let query):
            if state.phase != .stopping { state.query = query }
        case .folderSelected(let folder):
            if state.phase == .idle, !state.releasePending, state.execution == .standalone {
                state.folder = folder
                state.failure = nil
            }
        case .thresholdChanged(let value):
            if state.phase == .idle, value.isFinite, (0.3...0.9).contains(value) {
                state.minimumSimilarity = value
            }
        case .maximumBoxesChanged(let value):
            if state.phase == .idle, State.boxLimits.contains(value) { state.maximumBoxes = value }
        case .searchModeChanged(let mode):
            if state.phase == .idle { state.searchMode = mode }
        case .inputFailed(let message):
            if state.phase == .idle { state.failure = .operation(message) }
        case .poseChanged(let pose):
            let angle = state.orientation.angle
            state.orientation.update(pose)
            if angle != state.orientation.angle {
                state.orientationRevision += 1
                clear(&state)
                if state.phase == .starting || state.phase == .searching {
                    return [.cancel(.notification), .run(.orient(state.cameraRotation), id: .orientation)]
                }
            }
        case .start:
            guard state.canStart, let folder = state.folder else { return [] }
            state.session += 1
            state.submittedQuery = state.query.trimmingCharacters(in: .whitespacesAndNewlines)
            state.phase = .starting
            state.restartAfterStop = false
            clear(&state)
            state.failure = nil
            state.log = []
            state.analyzedFrames = 0
            state.peakFootprintBytes = nil
            return [
                .run(
                    .search(
                        state.session, folder, state.submittedQuery, state.minimumSimilarity, state.searchMode,
                        state.cameraRotation),
                    id: .search)
            ]
        case .applyQuery:
            guard state.canApplyQuery else { return [] }
            state.restartAfterStop = true
            state.phase = .stopping
            clear(&state)
            return [.cancel(.notification), .run(.stop(state.session, retainingModel: true), id: .control)]
        case .started(let session):
            guard state.session == session, state.phase == .starting, state.presented else { return [] }
            state.phase = .searching
        case .analyzed(let session, let frame):
            guard state.session == session, state.phase == .searching, state.presented else { return [] }
            guard frame.rotation == state.orientation.angle, frame.orientationRevision == state.orientationRevision
            else { return [] }
            let previous = state.matches.first
            state.frame = frame
            state.analyzedFrames += 1
            state.log = Array(([frame] + state.log).prefix(State.logLimit))
            if let footprint = frame.memory?.footprintBytes {
                state.peakFootprintBytes = max(state.peakFootprintBytes ?? 0, footprint)
            }
            state.matches = ObjectCandidateSelection.select(
                frame.candidates, minimumSimilarity: state.minimumSimilarity, limit: state.maximumBoxes)
            guard let first = state.matches.first else {
                state.stableFrames = 0
                state.emptyFrames += 1
                if state.emptyFrames >= 3 { state.notified = false }
                return [.cancel(.notification)]
            }
            state.emptyFrames = 0
            state.stableFrames = (previous?.region.overlap(with: first.region) ?? 0) >= 0.3 ? state.stableFrames + 1 : 1
            if state.stableFrames >= 2, !state.notified {
                state.notified = true
                return [.run(.notify(session), id: .notification)]
            }
        case .failed(let session, let failure):
            guard state.session == session, state.phase == .starting || state.phase == .searching else { return [] }
            state.phase = .idle
            state.restartAfterStop = false
            clear(&state)
            state.failure = failure
        case .stopFailed(let session, let failure):
            guard state.session == session, state.phase == .stopping || state.phase == .idle else { return [] }
            state.phase = .idle
            state.restartAfterStop = false
            state.releasePending = true
            clear(&state)
            state.failure = failure
        case .stop, .backgrounded, .disappeared:
            let cancelsReplacement = state.restartAfterStop
            state.restartAfterStop = false
            if case .disappeared = action { state.presented = false }
            clear(&state)
            guard state.phase != .idle, state.phase != .stopping else {
                if state.phase == .idle {
                    guard state.releasePending else { return [.cancelAll] }
                    state.phase = .stopping
                    state.failure = nil
                    return [.cancelAll, .run(.stop(state.session, retainingModel: false), id: .control)]
                }
                if cancelsReplacement {
                    return [.cancelAll, .run(.stop(state.session, retainingModel: false), id: .control)]
                }
                return []
            }
            state.phase = .stopping
            return [.cancelAll, .run(.stop(state.session, retainingModel: false), id: .control)]
        case .stopped(let session):
            guard state.session == session, state.phase == .stopping else { return [] }
            state.phase = .idle
            state.releasePending = false
            if state.restartAfterStop {
                state.restartAfterStop = false
                return reduce(&state, .start)
            }
        case .appeared:
            state.presented = true
        }
        return []
    }

    private static func clear(_ state: inout State) {
        state.frame = nil
        state.matches = []
        state.stableFrames = 0
        state.emptyFrames = 0
        state.notified = false
    }
}
