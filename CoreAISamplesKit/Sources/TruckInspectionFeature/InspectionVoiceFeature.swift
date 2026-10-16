import FeatureStore
import Foundation
import TruckInspectionCore

public enum InspectionPrompt: Equatable, Sendable {
    case question, confirmation, acknowledgment, completion
}

public enum InspectionVoiceAction: Sendable {
    case start, useButtons
    case prepared(Int, String)
    case spoken(Int, timedOut: Bool)
    case heard(Int, String)
    case partial(Int, String)
    case failed(Int, String)
    case clock(Int, Double)
}

public enum InspectionVoiceEffect: Equatable, Sendable {
    case prepare
    case speak(String)
    case listen([String])
}

extension TruckInspectionFeature {
    static func voiceReduce(_ state: inout State, _ action: InspectionVoiceAction) -> Commands {
        switch action {
        case .start:
            guard state.canConfigure else { return [] }
            guard state.modelInfo != nil else {
                state.message = "Prepare a model before starting voice input. Tap answers are available now."
                return []
            }
            state.generation += 1
            state.voiceActive = true
            state.phase = .authorizing
            state.speechTimeouts = 0
            state.emptyAnswers = 0
            state.unclearConfirmations = 0
            state.turnStartedAt = nil
            state.message = ""
            return [.run(.voice(state.generation, .prepare), id: .operation)]
        case .prepared(let id, let diagnostics):
            guard valid(state, id: id, phase: .authorizing) else { return [] }
            state.voiceDiagnostics = diagnostics
            state.note(diagnostics)
            next(&state)
            return ask(&state)
        case .spoken(let id, let timedOut):
            guard valid(state, id: id, phase: .speaking) else { return [] }
            if timedOut {
                state.speechTimeouts += 1
                if state.speechTimeouts >= 2 {
                    return voiceFallback(
                        &state, message: "Speech playback did not finish. Continue with the answer buttons.")
                }
                return speak(&state, text: state.spokenText, prompt: state.voicePrompt)
            }
            state.speechTimeouts = 0
            switch state.voicePrompt {
            case .question, .confirmation:
                state.generation += 1
                state.phase = .listening
                state.partialText = ""
                let hints = state.catalog.hints(for: state.currentItem, confirming: state.voicePrompt == .confirmation)
                return [.run(.voice(state.generation, .listen(hints)), id: .operation)]
            case .acknowledgment:
                state.turnStartedAt = nil
                state.emptyAnswers = 0
                state.unclearConfirmations = 0
                next(&state)
                return ask(&state)
            case .completion:
                return stop(&state, unload: false, resume: .completed)
            }
        case .heard(let id, let text):
            guard valid(state, id: id, phase: .listening) else { return [] }
            state.partialText = text
            state.note("final: \(text)")
            if state.voicePrompt == .confirmation {
                return confirmVoice(&state, text: text)
            }
            guard !InspectionRules.normalize(text).isEmpty else {
                state.emptyAnswers += 1
                if state.emptyAnswers >= 2 {
                    return voiceFallback(&state, message: "No answer was recognized. Continue with the answer buttons.")
                }
                return ask(&state, retry: true)
            }
            state.draft = text
            state.phase = .answering
            guard let item = state.currentItem else { return voiceFallback(&state, message: "") }
            let commands = submit(&state, item: item)
            if commands.isEmpty && state.phase == .answering { return voiceFallback(&state, message: state.message) }
            return commands
        case .partial(let id, let text):
            guard valid(state, id: id, phase: .listening), !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { return [] }
            state.partialText = text
            state.note("partial: \(text)")
            return []
        case .failed(let id, let error):
            guard state.active, state.voiceActive, id == state.generation else { return [] }
            state.note("voice-error: \(error)")
            return voiceFallback(
                &state,
                message:
                    "Voice input is unavailable. Check permissions and Japanese on-device recognition, or use the answer buttons."
            )
        case .useButtons:
            guard state.active, state.voiceActive else { return [] }
            return voiceFallback(&state, message: "Continue with the answer buttons.")
        case .clock(let id, let seconds):
            guard state.active, id == state.generation else { return [] }
            state.clockSeconds = seconds
            if state.voiceActive, state.phase == .speaking, state.voicePrompt == .question, state.turnStartedAt == nil {
                state.turnStartedAt = seconds
            }
            return []
        }
    }

    static func speak(_ state: inout State, text: String, prompt: InspectionPrompt) -> Commands {
        state.generation += 1
        state.phase = .speaking
        state.voicePrompt = prompt
        state.spokenText = text
        return [.run(.voice(state.generation, .speak(text)), id: .operation)]
    }

    static func ask(_ state: inout State, retry: Bool = false) -> Commands {
        guard let item = state.currentItem else {
            return speak(&state, text: state.catalog.vocabulary.completed, prompt: .completion)
        }
        return speak(&state, text: state.catalog.prompt(for: item, retry: retry), prompt: .question)
    }

    static func voiceFallback(_ state: inout State, message: String) -> Commands {
        state.message = message
        next(&state)
        let resume = state.phase
        return stop(&state, unload: false, resume: resume)
    }

    private static func valid(_ state: State, id: Int, phase: State.Phase) -> Bool {
        state.active && state.voiceActive && state.generation == id && state.phase == phase
    }

    private static func confirmVoice(_ state: inout State, text: String) -> Commands {
        switch InspectionRules.confirmation(text, vocabulary: state.catalog.vocabulary) {
        case .yes:
            state.phase = .confirming
            return reduce(&state, .confirm(true))
        case .no:
            state.phase = .confirming
            state.unclearConfirmations = 0
            return reduce(&state, .confirm(false))
        case .unknown:
            state.unclearConfirmations += 1
            guard state.unclearConfirmations < 2, let item = state.currentItem, let pending = state.pending else {
                return voiceFallback(
                    &state, message: "The result is still unconfirmed. Answer this item with the buttons.")
            }
            let text = state.catalog.confirmation(for: item, status: pending.status)
            return speak(&state, text: text, prompt: .confirmation)
        }
    }
}
