import FeatureStore
import Foundation
import Testing
import TruckInspectionCore
import TruckInspectionFeature
import TruckInspectionRuntime

struct InspectionVoiceTests {
    private func prepared() throws -> InspectionState {
        var state = InspectionState(catalog: try InspectionCatalogLoader.load())
        state.active = true
        state.modelInfo = InspectionModelInfo(
            folder: URL(filePath: "/asset"), revision: "test", milliseconds: 1, computePreference: "gpu")
        let start = TruckInspectionFeature.reduce(&state, .voice(.start))
        #expect(start == [.run(.voice(state.generation, .prepare), id: .operation)])
        #expect(state.phase == .authorizing)
        _ = TruckInspectionFeature.reduce(&state, .voice(.prepared(state.generation, "permissions and route ready")))
        #expect(state.phase == .speaking)
        return state
    }

    private func pending() throws -> InspectionState {
        var state = try prepared()
        _ = TruckInspectionFeature.reduce(&state, .voice(.spoken(state.generation, timedOut: false)))
        _ = TruckInspectionFeature.reduce(&state, .voice(.heard(state.generation, "A new rattling sound")))
        #expect(state.phase == .classifying)
        _ = TruckInspectionFeature.reduce(
            &state, .classified(state.generation, .init(status: .attention, method: .model, score: 1), 12))
        #expect(state.voicePrompt == .confirmation)
        _ = TruckInspectionFeature.reduce(&state, .voice(.spoken(state.generation, timedOut: false)))
        return state
    }

    @Test("TV1: Complete every inspection item by voice and measure question-to-confirmation time")
    func voiceRound() throws {
        var state = try prepared()
        let items = state.items
        for (index, item) in items.enumerated() {
            #expect(state.currentItem?.id == item.id)
            #expect(state.voicePrompt == .question)
            _ = TruckInspectionFeature.reduce(&state, .voice(.clock(state.generation, Double(index * 10))))
            let commands = TruckInspectionFeature.reduce(&state, .voice(.spoken(state.generation, timedOut: false)))
            #expect(
                commands == [
                    .run(
                        .voice(state.generation, .listen(state.catalog.hints(for: item, confirming: false))),
                        id: .operation)
                ])
            _ = TruckInspectionFeature.reduce(&state, .voice(.clock(state.generation, Double(index * 10 + 2))))
            let reply = state.catalog.title(for: .unchecked)
            let acknowledgment = TruckInspectionFeature.reduce(&state, .voice(.heard(state.generation, reply)))
            #expect(
                acknowledgment == [
                    .run(
                        .voice(state.generation, .speak(state.catalog.acknowledgment(for: item, status: .unchecked))),
                        id: .operation)
                ])
            #expect(state.records[item.id]?.candidate.status == .unchecked)
            #expect(state.records[item.id]?.turnMilliseconds == 2000)
            _ = TruckInspectionFeature.reduce(&state, .voice(.spoken(state.generation, timedOut: false)))
        }
        #expect(state.voicePrompt == .completion)
        _ = TruckInspectionFeature.reduce(&state, .voice(.spoken(state.generation, timedOut: false)))
        #expect(state.phase == .stopping)
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        #expect(state.phase == .completed)
        #expect(!state.voiceActive)
        #expect(state.records.count == items.count)
    }

    @Test("TV1: Negative confirmation repeats the full question and positive confirmation records the candidate")
    func confirmation() throws {
        var state = try pending()
        let item = try #require(state.currentItem)
        let no = try #require(state.catalog.vocabulary.no.first)
        let commands = TruckInspectionFeature.reduce(&state, .voice(.heard(state.generation, no)))
        #expect(
            commands == [
                .run(.voice(state.generation, .speak(state.catalog.prompt(for: item, retry: true))), id: .operation)
            ])
        #expect(state.records.isEmpty)
        #expect(state.pending == nil)
        state = try pending()
        let yes = try #require(state.catalog.vocabulary.yes.first)
        let accepted = TruckInspectionFeature.reduce(&state, .voice(.heard(state.generation, yes)))
        #expect(!accepted.isEmpty)
        #expect(state.voicePrompt == .acknowledgment)
        #expect(state.records[item.id]?.confirmed == true)
    }

    @Test("TV1: Two ambiguous confirmations leave the item unanswered and offer tap input")
    func uncertainConfirmation() throws {
        var state = try pending()
        _ = TruckInspectionFeature.reduce(&state, .voice(.heard(state.generation, "uncertain")))
        #expect(state.voicePrompt == .confirmation)
        _ = TruckInspectionFeature.reduce(&state, .voice(.spoken(state.generation, timedOut: false)))
        _ = TruckInspectionFeature.reduce(&state, .voice(.heard(state.generation, "")))
        #expect(state.phase == .stopping)
        #expect(state.records.isEmpty)
        #expect(state.pending == nil)
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        #expect(state.canAnswer)
    }

    @Test("TV1: Silence and repeated speech deadlines fall back without inventing an answer")
    func deadlines() throws {
        var silent = try prepared()
        for _ in 0..<2 {
            _ = TruckInspectionFeature.reduce(&silent, .voice(.spoken(silent.generation, timedOut: false)))
            _ = TruckInspectionFeature.reduce(&silent, .voice(.heard(silent.generation, "")))
        }
        #expect(silent.phase == .stopping)
        #expect(silent.records.isEmpty)
        var speaker = try prepared()
        _ = TruckInspectionFeature.reduce(&speaker, .voice(.spoken(speaker.generation, timedOut: true)))
        #expect(speaker.phase == .speaking)
        _ = TruckInspectionFeature.reduce(&speaker, .voice(.spoken(speaker.generation, timedOut: true)))
        #expect(speaker.phase == .stopping)
        #expect(!speaker.voiceActive)
        #expect(speaker.records.isEmpty)
    }

    @Test("TV1: Stale audio callbacks cannot change a stopped or restarted inspection")
    func staleAudio() throws {
        var state = try prepared()
        _ = TruckInspectionFeature.reduce(&state, .voice(.spoken(state.generation, timedOut: false)))
        let old = state.generation
        _ = TruckInspectionFeature.reduce(&state, .voice(.useButtons))
        let stopped = state
        for action: InspectionVoiceAction in [
            .start, .partial(old, "late"), .heard(old, "late"), .clock(old, 999), .spoken(old, timedOut: false),
            .failed(old, "late"),
        ] {
            #expect(TruckInspectionFeature.reduce(&state, .voice(action)).isEmpty)
            #expect(state == stopped)
        }
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        #expect(state.canAnswer)
        _ = TruckInspectionFeature.reduce(&state, .choose(.attention))
        #expect(state.records.count == 1)
        #expect(state.records.values.first?.turnMilliseconds == nil)
    }

    @Test("TV1: Voice setup failure leaves a usable tap question")
    func setupFailure() throws {
        var state = try prepared()
        _ = TruckInspectionFeature.reduce(&state, .voice(.failed(state.generation, "permission denied")))
        #expect(state.phase == .stopping)
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        #expect(state.canAnswer)
        #expect(!state.message.isEmpty)
        state.modelInfo = nil
        _ = TruckInspectionFeature.reduce(&state, .stop)
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        #expect(TruckInspectionFeature.reduce(&state, .voice(.start)).isEmpty)
        #expect(!state.voiceActive)
    }

    @Test(
        "INSPECT-02: Show the reason for a missing voice permission or capability, restrict only the affected operations, and keep typed inspection while the model is available",
        arguments: InspectionVoiceLimit.allCases)
    func voiceLimit(limit: InspectionVoiceLimit) throws {
        let folder = URL(filePath: "/library/artifacts/gliner/revision")
        var state = InspectionState(
            catalog: try InspectionCatalogLoader.load(), modelFolder: folder, execution: .managed,
            artifact: "gliner2.5-multi/variant@revision")
        _ = TruckInspectionFeature.reduce(&state, .appeared)
        state.modelInfo = InspectionModelInfo(
            folder: folder, revision: "test", milliseconds: 1, computePreference: "gpu")
        #expect(state.canStartVoice)
        _ = TruckInspectionFeature.reduce(&state, .voice(.start))
        #expect(state.phase == .authorizing)
        #expect(TruckInspectionFeature.reduce(&state, .voice(.limited(state.generation - 1, limit))).isEmpty)
        #expect(state.voiceLimit == nil)
        _ = TruckInspectionFeature.reduce(&state, .voice(.limited(state.generation, limit)))
        #expect(state.voiceLimit == limit)
        #expect(!limit.message.isEmpty)
        #expect(!state.voiceActive)
        #expect(state.phase == .stopping)
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        #expect(state.canAnswer)
        #expect(state.records.isEmpty)
        _ = TruckInspectionFeature.reduce(&state, .choose(.ok))
        #expect(state.records.count == 1)
        let item = try #require(state.currentItem)
        _ = TruckInspectionFeature.reduce(&state, .draftChanged("A new rattling sound"))
        #expect(
            TruckInspectionFeature.reduce(&state, .submit) == [
                .run(.classify(state.generation, "A new rattling sound", item, state.catalog), id: .operation)
            ])
        _ = TruckInspectionFeature.reduce(&state, .stop)
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        #expect(state.canConfigure)
        #expect(state.canMeasure)
        #expect(!state.canStartVoice)
        #expect(TruckInspectionFeature.reduce(&state, .voice(.start)).isEmpty)
        #expect(state.voiceLimit == limit)
        _ = TruckInspectionFeature.reduce(&state, .appeared)
        #expect(state.voiceLimit == nil)
        #expect(state.canStartVoice)
    }

    @Test("An acknowledgment fallback continues at the next unanswered item", arguments: [true, false])
    func acknowledgmentFallback(useButtons: Bool) throws {
        var state = try prepared()
        let nextID = state.items.dropFirst().first?.id
        _ = TruckInspectionFeature.reduce(&state, .voice(.spoken(state.generation, timedOut: false)))
        _ = TruckInspectionFeature.reduce(&state, .voice(.heard(state.generation, state.catalog.title(for: .ok))))
        #expect(state.voicePrompt == .acknowledgment)
        #expect(state.records.count == 1)
        if useButtons {
            _ = TruckInspectionFeature.reduce(&state, .voice(.useButtons))
        } else {
            _ = TruckInspectionFeature.reduce(&state, .voice(.spoken(state.generation, timedOut: true)))
            _ = TruckInspectionFeature.reduce(&state, .voice(.spoken(state.generation, timedOut: true)))
        }
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        #expect(state.canAnswer)
        #expect(state.currentItem?.id == nextID)
        #expect(state.records.count == 1)
    }
}
