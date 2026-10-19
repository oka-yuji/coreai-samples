import FeatureStore
import Foundation
import TruckInspectionCore

public enum InspectionAction: Sendable {
    case appeared, closed, start, stop, reset
    case voice(InspectionVoiceAction)
    case measure
    case measured(Int, InspectionMeasurement)
    case profileChanged(TruckProfile)
    case modelChosen(URL)
    case modelLoaded(Int, InspectionModelInfo)
    case draftChanged(String)
    case submit
    case choose(InspectionStatus)
    case classified(Int, InspectionCandidate, Double)
    case confirm(Bool)
    case edit(String)
    case failed(Int, String)
    case stopped(Int, unloaded: Bool)
    case stopFailed(Int, String)
    case diagnostic(Int, String)
}

public enum InspectionEffect: Equatable, Sendable {
    case voice(Int, InspectionVoiceEffect)
    case measure(Int, String, InspectionItem, InspectionCatalog)
    case prepare(Int, URL)
    case classify(Int, String, InspectionItem, InspectionCatalog)
    case stop(Int, unload: Bool)
}

public enum TruckInspectionFeature: Feature {
    public enum EffectID: Sendable { case operation, drain }
    public typealias State = InspectionState
    public typealias Action = InspectionAction
    public typealias Effect = InspectionEffect
    public typealias Commands = [Command<Effect, EffectID>]

    public static func reduce(_ state: inout State, _ action: Action) -> Commands {
        switch action {
        case .voice(let voice): return voiceReduce(&state, voice)
        case .measure:
            guard state.canConfigure, let item = state.items.first(where: { $0.id == "tire_pressure" }),
                let text = item.aliases[InspectionStatus.attention.rawValue]?.first
            else { return [] }
            guard state.modelInfo != nil else { return prepareFirst(&state, .measure) ?? [] }
            state.generation += 1
            state.measurement = nil
            state.message = ""
            state.measurementReturnPhase = state.phase
            state.phase = .measuring
            return [.run(.measure(state.generation, text, item, state.catalog), id: .operation)]
        case .measured(let id, let result):
            guard state.active, id == state.generation, state.phase == .measuring else { return [] }
            state.measurement = result
            state.phase = state.measurementReturnPhase
            state.note(
                "benchmark: runs=20 warmupMs=\(result.warmupMilliseconds) medianMs=\(result.medianMilliseconds) p95Ms=\(result.p95Milliseconds)"
            )
            return []
        case .appeared:
            state.active = true
            state.voiceLimit = nil
            guard state.execution == .standalone else { return [] }
            if state.phase == .stopping {
                state.reloadAfterStop = true
                return []
            }
            guard state.phase == .idle, state.modelInfo == nil, let folder = state.modelFolder else { return [] }
            return prepare(&state, folder: folder)
        case .closed:
            guard state.active || state.phase == .stopFailed else { return [] }
            state.active = false
            state.reloadAfterStop = false
            return stop(&state, unload: true)
        case .modelChosen(let folder):
            guard state.execution == .standalone, state.canConfigure else { return [] }
            return prepare(&state, folder: folder)
        case .modelLoaded(let id, let info):
            guard state.active, id == state.generation, state.phase == .preparing else { return [] }
            state.modelInfo = info
            state.phase = state.requestPhase
            state.note(
                "timing: load=\(info.milliseconds)ms revision=\(info.revision) preference=\(info.computePreference) floatType=\(info.floatType)"
            )
            guard let request = state.request else { return [] }
            state.request = nil
            return resume(&state, request)
        case .profileChanged(let profile):
            guard state.canConfigure, state.profile != profile else { return [] }
            state.profile = profile
            reset(&state)
            return []
        case .start:
            guard state.canConfigure else { return [] }
            if let commands = prepareFirst(&state, .start) { return commands }
            state.message = ""
            next(&state)
            return []
        case .reset:
            guard state.canConfigure else { return [] }
            reset(&state)
            return []
        case .draftChanged(let text):
            guard state.canAnswer else { return [] }
            state.draft = text
            state.message = ""
            return []
        case .submit:
            guard state.canAnswer, let item = state.currentItem else { return [] }
            return submit(&state, item: item)
        case .choose(let status):
            guard state.canAnswer else { return [] }
            state.draft = state.catalog.title(for: status)
            return record(&state, candidate: InspectionCandidate(status: status, method: .button), confirmed: false)
        case .classified(let id, let candidate, let milliseconds):
            guard state.active, id == state.generation, state.phase == .classifying else { return [] }
            state.pending = candidate
            state.pendingMilliseconds = milliseconds
            state.phase = .confirming
            state.note("timing: classify=\(milliseconds)ms score=\(candidate.score ?? 0)")
            if state.voiceActive, let item = state.currentItem {
                return speak(
                    &state, text: state.catalog.confirmation(for: item, status: candidate.status), prompt: .confirmation
                )
            }
            return []
        case .confirm(let accepted):
            guard state.active, state.phase == .confirming, let pending = state.pending else { return [] }
            if accepted {
                return record(&state, candidate: pending, confirmed: true)
            } else {
                state.pending = nil
                state.phase = .answering
                state.message = "Please answer this item again."
                if state.voiceActive { return ask(&state, retry: true) }
            }
            return []
        case .edit(let id):
            guard state.canConfigure, let index = state.items.firstIndex(where: { $0.id == id }) else { return [] }
            state.records[id] = nil
            state.index = index
            state.phase = .answering
            state.draft = ""
            state.pending = nil
            state.message = ""
            return []
        case .stop:
            guard state.active, state.phase != .stopping else { return [] }
            return stop(&state, unload: false)
        case .stopped(let id, let unloaded):
            guard id == state.generation, state.phase == .stopping else { return [] }
            state.phase = state.active ? state.resumeAfterStop : .idle
            if unloaded { state.modelInfo = nil }
            let reload = state.reloadAfterStop
            state.reloadAfterStop = false
            if reload, state.active, state.modelInfo == nil, let folder = state.modelFolder {
                return prepare(&state, folder: folder)
            }
            return []
        case .stopFailed(let id, let error):
            guard id == state.generation, state.phase == .stopping else { return [] }
            state.phase = .stopFailed
            state.note("stop-error: \(error)")
            return []
        case .failed(let id, let error):
            guard state.active, id == state.generation,
                state.phase == .preparing || state.phase == .classifying || state.phase == .measuring
            else { return [] }
            let request = state.request
            state.request = nil
            state.phase =
                state.phase == .measuring
                ? state.measurementReturnPhase : (state.phase == .preparing ? state.requestPhase : .answering)
            state.message = "The operation failed. Check the diagnostics or use the answer buttons."
            state.note("error: \(error)")
            if request == .start { next(&state) }
            if state.voiceActive { return voiceFallback(&state, message: state.message) }
            return []
        case .diagnostic(let id, let text):
            guard state.active, id == state.generation else { return [] }
            state.note(text)
            return []
        }
    }

    static func prepare(_ state: inout State, folder: URL) -> Commands {
        state.generation += 1
        state.modelFolder = folder
        state.modelInfo = nil
        state.measurement = nil
        state.phase = .preparing
        state.message = ""
        state.request = nil
        state.requestPhase = .idle
        return [.run(.prepare(state.generation, folder), id: .operation)]
    }

    static func submit(_ state: inout State, item: InspectionItem) -> Commands {
        do {
            if let exact = try InspectionRules.exactAnswer(state.draft, item: item, catalog: state.catalog) {
                return record(&state, candidate: exact, confirmed: false)
            }
        } catch {
            state.message =
                error as? InspectionError == .answerTooLong
                ? "Keep the answer within 240 characters." : "Enter an answer first."
            return []
        }
        guard state.modelInfo != nil else {
            if let commands = prepareFirst(&state, .submit) { return commands }
            state.message = "Choose a model asset to interpret free-form answers, or use the answer buttons."
            return []
        }
        state.generation += 1
        state.phase = .classifying
        state.message = ""
        return [.run(.classify(state.generation, state.draft, item, state.catalog), id: .operation)]
    }

    static func record(_ state: inout State, candidate: InspectionCandidate, confirmed: Bool) -> Commands {
        guard let item = state.currentItem else { return [] }
        state.records[item.id] = InspectionRecord(
            itemID: item.id, title: item.title, answer: state.draft, candidate: candidate,
            confirmed: confirmed, milliseconds: candidate.method == .model ? state.pendingMilliseconds : 0,
            turnMilliseconds: state.voiceActive
                ? state.turnStartedAt.map { max(0, state.clockSeconds - $0) * 1000 } : nil,
            artifact: candidate.method == .model ? state.artifact : nil)
        state.note(
            "decided: \(item.id)=\(candidate.status.rawValue) via=\(candidate.method.rawValue) confirmed=\(confirmed)")
        state.pending = nil
        state.pendingMilliseconds = 0
        state.draft = ""
        state.message = ""
        if state.voiceActive {
            if let start = state.turnStartedAt { state.note("timing: turn=\(max(0, state.clockSeconds - start))s") }
            return speak(
                &state, text: state.catalog.acknowledgment(for: item, status: candidate.status), prompt: .acknowledgment
            )
        }
        next(&state)
        return []
    }

    static func next(_ state: inout State) {
        if let next = state.items.firstIndex(where: { state.records[$0.id] == nil }) {
            state.index = next
            state.phase = .answering
        } else {
            state.index = state.items.count
            state.phase = .completed
        }
    }

    static func reset(_ state: inout State) {
        state.generation += 1
        state.records.removeAll()
        state.events.removeAll()
        state.index = 0
        state.pending = nil
        state.draft = ""
        state.message = ""
        state.phase = .idle
        state.partialText = ""
        state.turnStartedAt = nil
    }

    static func stop(_ state: inout State, unload: Bool, resume: State.Phase = .idle) -> Commands {
        state.voiceActive = false
        state.resumeAfterStop = resume
        state.generation += 1
        state.pending = nil
        state.request = nil
        state.phase = .stopping
        return [.cancelAll, .run(.stop(state.generation, unload: unload), id: .drain)]
    }
}

extension TruckInspectionFeature {
    static func prepareFirst(_ state: inout State, _ request: State.Request) -> Commands? {
        guard state.preparesOnRequest, state.modelInfo == nil, let folder = state.modelFolder else { return nil }
        let phase = state.phase
        let commands = prepare(&state, folder: folder)
        state.request = request
        state.requestPhase = phase
        return commands
    }

    static func resume(_ state: inout State, _ request: State.Request) -> Commands {
        switch request {
        case .start: reduce(&state, .start)
        case .submit: reduce(&state, .submit)
        case .measure: reduce(&state, .measure)
        case .voice: reduce(&state, .voice(.start))
        }
    }
}
