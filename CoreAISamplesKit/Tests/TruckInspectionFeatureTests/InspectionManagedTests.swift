import Foundation
import Testing
import TruckInspectionCore
import TruckInspectionFeature
import TruckInspectionRuntime

struct InspectionManagedTests {
    private let artifactFolder = URL(filePath: "/library/artifacts/gliner/revision")

    private func managed(artifact: String = "gliner2.5-multi/variant@revision") throws -> InspectionState {
        var state = InspectionState(
            catalog: try InspectionCatalogLoader.load(), modelFolder: artifactFolder, execution: .managed,
            artifact: artifact)
        #expect(TruckInspectionFeature.reduce(&state, .appeared).isEmpty)
        return state
    }

    private func info() -> InspectionModelInfo {
        InspectionModelInfo(folder: artifactFolder, revision: "test", milliseconds: 1, computePreference: "gpu")
    }

    @Test(
        "INSPECT-01: Inspect with only the selected artifact and start no preparation, recording, speech or permission request while browsing or after download completes"
    )
    func managedPreparation() throws {
        var state = try managed()
        #expect(state.active)
        #expect(state.phase == .idle)
        #expect(state.modelInfo == nil)
        #expect(TruckInspectionFeature.reduce(&state, .modelChosen(URL(filePath: "/other"))).isEmpty)
        #expect(state.modelFolder == artifactFolder)
        #expect(
            TruckInspectionFeature.reduce(&state, .start) == [
                .run(.prepare(state.generation, artifactFolder), id: .operation)
            ])
        #expect(TruckInspectionFeature.reduce(&state, .start).isEmpty)
        #expect(TruckInspectionFeature.reduce(&state, .modelLoaded(state.generation, info())).isEmpty)
        #expect(state.phase == .answering)
        #expect(state.currentItem?.id == state.items.first?.id)
        let item = try #require(state.currentItem)
        _ = TruckInspectionFeature.reduce(&state, .draftChanged("A new rattling sound"))
        #expect(
            TruckInspectionFeature.reduce(&state, .submit) == [
                .run(.classify(state.generation, "A new rattling sound", item, state.catalog), id: .operation)
            ])
        _ = TruckInspectionFeature.reduce(&state, .closed)
        _ = TruckInspectionFeature.reduce(&state, .appeared)
        #expect(!state.reloadAfterStop)
        #expect(TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: true)).isEmpty)
        #expect(state.phase == .idle)
        #expect(state.modelInfo == nil)
        #expect(TruckInspectionFeature.reduce(&state, .appeared).isEmpty)
    }

    @Test("INSPECT-01: An explicit typed answer, measurement or voice start prepares the model once and continues")
    func managedRequests() throws {
        var typed = try managed()
        typed.phase = .answering
        let item = try #require(typed.currentItem)
        _ = TruckInspectionFeature.reduce(&typed, .draftChanged("A new rattling sound"))
        #expect(
            TruckInspectionFeature.reduce(&typed, .submit) == [
                .run(.prepare(typed.generation, artifactFolder), id: .operation)
            ])
        #expect(typed.draft == "A new rattling sound")
        #expect(
            TruckInspectionFeature.reduce(&typed, .modelLoaded(typed.generation, info())) == [
                .run(.classify(typed.generation, "A new rattling sound", item, typed.catalog), id: .operation)
            ])
        #expect(typed.phase == .classifying)

        var measured = try managed()
        _ = TruckInspectionFeature.reduce(&measured, .measure)
        let resumed = TruckInspectionFeature.reduce(&measured, .modelLoaded(measured.generation, info()))
        guard case .run(.measure, _)? = resumed.first else {
            Issue.record("Expected the measurement to continue after preparation")
            return
        }
        #expect(measured.phase == .measuring)
        #expect(measured.measurementReturnPhase == .idle)

        var voice = try managed()
        _ = TruckInspectionFeature.reduce(&voice, .voice(.start))
        #expect(voice.phase == .preparing)
        #expect(!voice.voiceActive)
        #expect(
            TruckInspectionFeature.reduce(&voice, .modelLoaded(voice.generation, info())) == [
                .run(.voice(voice.generation, .prepare), id: .operation)
            ])
        #expect(voice.phase == .authorizing)
    }

    @Test("INSPECT-01: A failed managed preparation keeps tap answers available and a later request retries it")
    func managedPreparationFailure() throws {
        var state = try managed()
        _ = TruckInspectionFeature.reduce(&state, .start)
        _ = TruckInspectionFeature.reduce(&state, .failed(state.generation, "load failed"))
        #expect(state.phase == .answering)
        #expect(state.modelInfo == nil)
        #expect(!state.message.isEmpty)
        _ = TruckInspectionFeature.reduce(&state, .choose(.ok))
        #expect(state.records.count == 1)
        _ = TruckInspectionFeature.reduce(&state, .draftChanged("Something sounds different"))
        #expect(
            TruckInspectionFeature.reduce(&state, .submit) == [
                .run(.prepare(state.generation, artifactFolder), id: .operation)
            ])
    }

    @Test("INSPECT-03: A failed stop blocks new work until a retried stop succeeds and releases the model")
    func stopFailure() throws {
        var state = try managed()
        _ = TruckInspectionFeature.reduce(&state, .start)
        _ = TruckInspectionFeature.reduce(&state, .modelLoaded(state.generation, info()))
        #expect(
            TruckInspectionFeature.reduce(&state, .stop) == [
                .cancelAll, .run(.stop(state.generation, unload: false), id: .drain),
            ])
        #expect(TruckInspectionFeature.reduce(&state, .stopFailed(state.generation - 1, "stale")).isEmpty)
        #expect(state.phase == .stopping)
        _ = TruckInspectionFeature.reduce(&state, .stopFailed(state.generation, "deactivation failed"))
        #expect(state.phase == .stopFailed)
        #expect(!state.message.isEmpty)
        #expect(state.events.last == "stop-error: deactivation failed")
        let failed = state
        let item = try #require(state.items.first)
        for action: InspectionAction in [
            .start, .reset, .choose(.ok), .submit, .measure, .voice(.start), .edit(item.id),
            .profileChanged(.hydraulic), .modelChosen(URL(filePath: "/other")),
            .stopped(state.generation - 1, unloaded: true),
        ] {
            #expect(TruckInspectionFeature.reduce(&state, action).isEmpty)
            #expect(state == failed)
        }
        #expect(
            TruckInspectionFeature.reduce(&state, .stop) == [
                .cancelAll, .run(.stop(state.generation, unload: false), id: .drain),
            ])
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        #expect(state.phase == .idle)
        #expect(TruckInspectionFeature.reduce(&state, .start).isEmpty)
        #expect(state.phase == .answering)

        #expect(
            TruckInspectionFeature.reduce(&state, .closed) == [
                .cancelAll, .run(.stop(state.generation, unload: true), id: .drain),
            ])
        _ = TruckInspectionFeature.reduce(&state, .stopFailed(state.generation, "deactivation failed"))
        #expect(state.phase == .stopFailed)
        #expect(!state.active)
        #expect(
            TruckInspectionFeature.reduce(&state, .closed) == [
                .cancelAll, .run(.stop(state.generation, unload: true), id: .drain),
            ])
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: true))
        #expect(state.phase == .idle)
        #expect(state.modelInfo == nil)
        #expect(TruckInspectionFeature.reduce(&state, .closed).isEmpty)
    }

    @Test(
        "INSPECT-04: Keep inspection answers, corrections, records and their artifact attribution, and never apply stale recognition or classification results to another question or demo"
    )
    func attributionAndRetention() throws {
        var state = try managed(artifact: "gliner2.5-multi/variant-a@revision-a")
        _ = TruckInspectionFeature.reduce(&state, .start)
        _ = TruckInspectionFeature.reduce(&state, .modelLoaded(state.generation, info()))
        let first = try #require(state.currentItem)
        _ = TruckInspectionFeature.reduce(&state, .draftChanged("The pedal feels softer than usual"))
        _ = TruckInspectionFeature.reduce(&state, .submit)
        let candidate = InspectionCandidate(status: .attention, method: .model, score: 0.9)
        _ = TruckInspectionFeature.reduce(&state, .classified(state.generation, candidate, 12))
        _ = TruckInspectionFeature.reduce(&state, .confirm(true))
        #expect(state.records[first.id]?.artifact == "gliner2.5-multi/variant-a@revision-a")
        let second = try #require(state.currentItem)
        _ = TruckInspectionFeature.reduce(&state, .choose(.ok))
        #expect(state.records[second.id]?.artifact == nil)
        let third = try #require(state.currentItem)
        _ = TruckInspectionFeature.reduce(&state, .draftChanged("Another unusual noise"))
        _ = TruckInspectionFeature.reduce(&state, .submit)
        let late = state.generation
        _ = TruckInspectionFeature.reduce(&state, .stop)
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        _ = TruckInspectionFeature.reduce(&state, .edit(second.id))
        _ = TruckInspectionFeature.reduce(&state, .choose(.attention))
        #expect(state.records[second.id]?.candidate.status == .attention)
        _ = TruckInspectionFeature.reduce(&state, .classified(late, candidate, 1))
        _ = TruckInspectionFeature.reduce(&state, .voice(.heard(late, "late answer")))
        #expect(state.records[third.id] == nil)
        #expect(state.pending == nil)
        #expect(state.currentItem?.id == third.id)
        _ = TruckInspectionFeature.reduce(&state, .draftChanged("Draft kept for this question"))
        _ = TruckInspectionFeature.reduce(&state, .closed)
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: true))
        #expect(state.report.contains("model: gliner2.5-multi/variant-a@revision-a"))

        var next = InspectionState(
            catalog: state.catalog, modelFolder: URL(filePath: "/library/artifacts/gliner/revision-b"),
            execution: .managed, artifact: "gliner2.5-multi/variant-b@revision-b")
        next.retain(state)
        #expect(next.records == state.records)
        #expect(next.profile == state.profile)
        #expect(next.draft == "Draft kept for this question")
        #expect(next.events == state.events)
        #expect(next.modelInfo == nil)
        #expect(next.phase == .idle)
        #expect(!next.active)
        #expect(next.pending == nil)
        #expect(next.message.isEmpty)
        #expect(next.artifact == "gliner2.5-multi/variant-b@revision-b")
        _ = TruckInspectionFeature.reduce(&next, .appeared)
        _ = TruckInspectionFeature.reduce(&next, .classified(late, candidate, 1))
        #expect(next.records == state.records)
        #expect(next.records[first.id]?.artifact == "gliner2.5-multi/variant-a@revision-a")
    }
}
