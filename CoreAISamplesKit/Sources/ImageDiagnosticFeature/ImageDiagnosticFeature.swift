import FeatureStore
import Foundation
import ImageGenerationCore

public struct ImageDiagnosticStop: Codable, Equatable, Sendable {
    public let component: ImageModelComponent
    public let operation: ImageMeasuredOperation
    public let step: Int?
    public let delayMilliseconds: Int

    public func matches(_ phase: ImageExecutionPhase) -> Bool {
        switch phase {
        case .loading(let value): operation == .loading && value == component && step == nil
        case .running(let value, let index): operation == .inference && value == component && index == step
        default: false
        }
    }
}

public struct ImageDiagnosticOutcome: Equatable, Sendable {
    public let status: ImageExecutionReport.Status
    public let directory: String
    public let message: String

    public init(status: ImageExecutionReport.Status, directory: String, message: String) {
        self.status = status
        self.directory = directory
        self.message = message
    }
}

public struct ImageDiagnosticState: Equatable, Sendable {
    public enum Phase: Equatable, Sendable { case idle, running, stopping, cleanupFailed, finished }
    public var phase: Phase = .idle
    public var generation = 0
    public var message = "Ready"
    public var outcome: ImageDiagnosticOutcome?
    public var stopPlan: ImageDiagnosticStop?
    public var stopScheduled = false
    public init() {}
}

public enum ImageDiagnosticAction: Sendable {
    case start(URL)
    case configured(Int, ImageDiagnosticStop?)
    case progressed(Int, ImageExecutionPhase)
    case completed(Int, ImageDiagnosticOutcome)
    case failed(Int, String)
    case stop, ended
    case stopElapsed(Int)
    case stopped(Int, ImageDiagnosticOutcome?)
    case stopFailed(Int, String)
}

public enum ImageDiagnosticEffect: Equatable, Sendable {
    case execute(Int, URL)
    case stop(Int)
    case delayStop(Int, Int)
}

public enum ImageDiagnosticEffectID: Hashable, Sendable { case execution, stop, delay }

public enum ImageDiagnosticFeature: Feature {
    public typealias State = ImageDiagnosticState
    public typealias Action = ImageDiagnosticAction
    public typealias Effect = ImageDiagnosticEffect
    public typealias EffectID = ImageDiagnosticEffectID
    public typealias Command = FeatureStore.Command<Effect, EffectID>

    public static func reduce(_ state: inout State, _ action: Action) -> [Command] {
        switch action {
        case .start(let url):
            guard state.phase == .idle else { return [] }
            state.generation += 1
            state.phase = .running
            state.message = "Preparing diagnostic request"
            return [.run(.execute(state.generation, url), id: .execution)]
        case .configured(let generation, let plan):
            guard generation == state.generation, state.phase == .running else { return [] }
            state.stopPlan = plan
        case .progressed(let generation, let phase):
            guard generation == state.generation, state.phase == .running else { return [] }
            state.message = String(describing: phase)
            if !state.stopScheduled, let plan = state.stopPlan, plan.matches(phase) {
                state.stopScheduled = true
                return [.run(.delayStop(generation, plan.delayMilliseconds), id: .delay)]
            }
        case .completed(let generation, let outcome):
            guard generation == state.generation, state.phase == .running else { return [] }
            state.outcome = outcome
            state.message = outcome.message
            state.phase = outcome.status == .releaseUnconfirmed ? .cleanupFailed : .finished
            return [.cancel(.delay)]
        case .failed(let generation, let message):
            guard generation == state.generation, state.phase == .running else { return [] }
            state.phase = .finished
            state.message = message
            return [.cancel(.delay)]
        case .stopElapsed(let generation):
            guard generation == state.generation, state.phase == .running else { return [] }
            return beginStop(&state)
        case .stop:
            guard [.running, .cleanupFailed].contains(state.phase) else { return [] }
            return beginStop(&state)
        case .ended:
            if state.phase == .stopping { return [] }
            guard [.running, .cleanupFailed].contains(state.phase) else { return [.cancelAll] }
            return beginStop(&state)
        case .stopped(let generation, let outcome):
            guard generation == state.generation, state.phase == .stopping else { return [] }
            state.outcome =
                outcome
                ?? ImageDiagnosticOutcome(status: .cancelled, directory: "", message: "Diagnostic request stopped")
            state.phase = .finished
            state.message = outcome?.message ?? "Diagnostic request stopped"
        case .stopFailed(let generation, let message):
            guard generation == state.generation, state.phase == .stopping else { return [] }
            state.phase = .cleanupFailed
            state.message = message
        }
        return []
    }

    private static func beginStop(_ state: inout State) -> [Command] {
        state.generation += 1
        state.phase = .stopping
        state.message = "Stopping and joining diagnostic request"
        return [.cancelAll, .run(.stop(state.generation), id: .stop)]
    }
}
