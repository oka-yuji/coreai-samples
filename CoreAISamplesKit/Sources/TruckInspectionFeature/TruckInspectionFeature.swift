import FeatureStore
import Foundation
import TruckInspectionCore

public enum InspectionAction: Sendable {
    case appeared, closed, start, stop, reset
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
    case diagnostic(Int, String)
}

public enum InspectionEffect: Equatable, Sendable {
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
        case .appeared:
            state.active = true
            if state.phase == .stopping {
                state.reloadAfterStop = true
                return []
            }
            guard state.phase == .idle, state.modelInfo == nil, let folder = state.modelFolder else { return [] }
            return prepare(&state, folder: folder)
        case .closed:
            guard state.active else { return [] }
            state.active = false
            state.reloadAfterStop = false
            return stop(&state, unload: true)
        case .modelChosen(let folder):
            guard state.canConfigure else { return [] }
            return prepare(&state, folder: folder)
        case .modelLoaded(let id, let info):
            guard state.active, id == state.generation, state.phase == .preparing else { return [] }
            state.modelInfo = info
            state.phase = .idle
            state.note(
                "timing: load=\(info.milliseconds)ms revision=\(info.revision) preference=\(info.computePreference)")
            return []
        case .profileChanged(let profile):
            guard state.canConfigure, state.profile != profile else { return [] }
            state.profile = profile
            reset(&state)
            return []
        case .start:
            guard state.canConfigure else { return [] }
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
            record(&state, candidate: InspectionCandidate(status: status, method: .button), confirmed: false)
            return []
        case .classified(let id, let candidate, let milliseconds):
            guard state.active, id == state.generation, state.phase == .classifying else { return [] }
            state.pending = candidate
            state.pendingMilliseconds = milliseconds
            state.phase = .confirming
            state.note("timing: classify=\(milliseconds)ms score=\(candidate.score ?? 0)")
            return []
        case .confirm(let accepted):
            guard state.active, state.phase == .confirming, let pending = state.pending else { return [] }
            if accepted {
                record(&state, candidate: pending, confirmed: true)
            } else {
                state.pending = nil
                state.phase = .answering
                state.message = "Please answer this item again."
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
            state.phase = .idle
            if unloaded { state.modelInfo = nil }
            let reload = state.reloadAfterStop
            state.reloadAfterStop = false
            if reload, state.active, state.modelInfo == nil, let folder = state.modelFolder {
                return prepare(&state, folder: folder)
            }
            return []
        case .failed(let id, let error):
            guard state.active, id == state.generation,
                state.phase == .preparing || state.phase == .classifying
            else { return [] }
            state.phase = state.phase == .preparing ? .idle : .answering
            state.message = "The operation failed. Check the diagnostics or use the answer buttons."
            state.note("error: \(error)")
            return []
        case .diagnostic(let id, let text):
            guard state.active, id == state.generation else { return [] }
            state.note(text)
            return []
        }
    }

    private static func prepare(_ state: inout State, folder: URL) -> Commands {
        state.generation += 1
        state.modelFolder = folder
        state.modelInfo = nil
        state.phase = .preparing
        state.message = ""
        return [.run(.prepare(state.generation, folder), id: .operation)]
    }

    private static func submit(_ state: inout State, item: InspectionItem) -> Commands {
        do {
            if let exact = try InspectionRules.exactAnswer(state.draft, item: item, catalog: state.catalog) {
                record(&state, candidate: exact, confirmed: false)
                return []
            }
        } catch {
            state.message =
                error as? InspectionError == .answerTooLong
                ? "Keep the answer within 240 characters." : "Enter an answer first."
            return []
        }
        guard state.modelInfo != nil else {
            state.message = "Choose a model asset to interpret free-form answers, or use the answer buttons."
            return []
        }
        state.generation += 1
        state.phase = .classifying
        state.message = ""
        return [.run(.classify(state.generation, state.draft, item, state.catalog), id: .operation)]
    }

    private static func record(_ state: inout State, candidate: InspectionCandidate, confirmed: Bool) {
        guard let item = state.currentItem else { return }
        state.records[item.id] = InspectionRecord(
            itemID: item.id, title: item.title, answer: state.draft, candidate: candidate,
            confirmed: confirmed, milliseconds: candidate.method == .model ? state.pendingMilliseconds : 0)
        state.note(
            "decided: \(item.id)=\(candidate.status.rawValue) via=\(candidate.method.rawValue) confirmed=\(confirmed)")
        state.pending = nil
        state.pendingMilliseconds = 0
        state.draft = ""
        state.message = ""
        next(&state)
    }

    private static func next(_ state: inout State) {
        if let next = state.items.firstIndex(where: { state.records[$0.id] == nil }) {
            state.index = next
            state.phase = .answering
        } else {
            state.index = state.items.count
            state.phase = .completed
        }
    }

    private static func reset(_ state: inout State) {
        state.generation += 1
        state.records.removeAll()
        state.events.removeAll()
        state.index = 0
        state.pending = nil
        state.draft = ""
        state.message = ""
        state.phase = .idle
    }

    private static func stop(_ state: inout State, unload: Bool) -> Commands {
        state.generation += 1
        state.pending = nil
        state.phase = .stopping
        return [.cancelAll, .run(.stop(state.generation, unload: unload), id: .drain)]
    }
}
