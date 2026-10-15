import FeatureStore
import Foundation
import Testing
import TruckInspectionCore
import TruckInspectionFeature
import TruckInspectionRuntime

struct InspectionFeatureTests {
    @Test("Truck model selection excludes unrelated assets in the shared Documents folder")
    func modelFamilies() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let gliner = root.appending(path: "gliner")
        let embedding = root.appending(path: "embedding")
        let broken = root.appending(path: "broken")
        for folder in [gliner, embedding, broken] {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        }
        try Data(#"{"model":"fastino/gliner2.5-multi-v1"}"#.utf8).write(to: gliner.appending(path: "contract.json"))
        try Data(#"{"model":"google/embeddinggemma2"}"#.utf8).write(to: embedding.appending(path: "contract.json"))
        try Data("invalid".utf8).write(to: broken.appending(path: "contract.json"))
        #expect(InspectionCatalogLoader.modelFolders(in: [embedding, broken, gliner]) == [gliner])
    }

    private func state(model: Bool = false) throws -> InspectionState {
        var state = InspectionState(catalog: try InspectionCatalogLoader.load())
        state.active = true
        if model {
            state.modelInfo = InspectionModelInfo(
                folder: URL(filePath: "/asset"), revision: "test", milliseconds: 1, computePreference: "gpu")
        }
        return state
    }

    @Test("TT1: Complete and correct a truck inspection one question at a time")
    func completeAndEdit() throws {
        var state = try state()
        #expect(TruckInspectionFeature.reduce(&state, .start).isEmpty)
        let items = state.items
        for item in items {
            #expect(state.currentItem?.id == item.id)
            #expect(state.phase == .answering)
            #expect(TruckInspectionFeature.reduce(&state, .choose(.unchecked)).isEmpty)
            #expect(state.records[item.id]?.candidate.status == .unchecked)
        }
        #expect(state.phase == .completed)
        #expect(state.unansweredCount == 0)
        let first = try #require(items.first)
        #expect(TruckInspectionFeature.reduce(&state, .edit(first.id)).isEmpty)
        #expect(state.currentItem?.id == first.id)
        #expect(state.records.count == items.count - 1)
        _ = TruckInspectionFeature.reduce(&state, .choose(.attention))
        #expect(state.phase == .completed)
        #expect(state.records[first.id]?.candidate.status == .attention)
        #expect(state.report.contains(first.title))
    }

    @Test("TT1: Exact answers precede classification and inferred answers require confirmation")
    func exactAndModel() throws {
        var state = try state(model: true)
        _ = TruckInspectionFeature.reduce(&state, .start)
        _ = TruckInspectionFeature.reduce(&state, .draftChanged(state.catalog.title(for: .ok)))
        #expect(TruckInspectionFeature.reduce(&state, .submit).isEmpty)
        #expect(state.records.count == 1)
        let item = try #require(state.currentItem)
        let answer = "The pedal feels softer than usual"
        _ = TruckInspectionFeature.reduce(&state, .draftChanged(answer))
        let commands = TruckInspectionFeature.reduce(&state, .submit)
        #expect(commands == [.run(.classify(state.generation, answer, item, state.catalog), id: .operation)])
        #expect(state.phase == .classifying)
        let candidate = InspectionCandidate(status: .attention, method: .model, score: 1)
        _ = TruckInspectionFeature.reduce(&state, .classified(state.generation, candidate, 12))
        #expect(state.phase == .confirming)
        #expect(state.records[item.id] == nil)
        _ = TruckInspectionFeature.reduce(&state, .confirm(false))
        #expect(state.phase == .answering)
        #expect(state.pending == nil)
        _ = TruckInspectionFeature.reduce(&state, .submit)
        _ = TruckInspectionFeature.reduce(&state, .classified(state.generation, candidate, 15))
        _ = TruckInspectionFeature.reduce(&state, .confirm(true))
        #expect(state.records[item.id]?.confirmed == true)
        #expect(state.records[item.id]?.milliseconds == 15)
    }

    @Test("TT2: Ignore old completions and reject restart until cleanup finishes")
    func stopAndStaleCompletion() throws {
        var state = try state(model: true)
        _ = TruckInspectionFeature.reduce(&state, .start)
        _ = TruckInspectionFeature.reduce(&state, .draftChanged("A different sound"))
        _ = TruckInspectionFeature.reduce(&state, .submit)
        let old = state.generation
        let stop = TruckInspectionFeature.reduce(&state, .stop)
        #expect(stop == [.cancelAll, .run(.stop(state.generation, unload: false), id: .drain)])
        let stopping = state
        for action: InspectionAction in [.start, .reset, .modelChosen(URL(filePath: "/other")), .choose(.ok), .stop] {
            #expect(TruckInspectionFeature.reduce(&state, action).isEmpty)
            #expect(state == stopping)
        }
        _ = TruckInspectionFeature.reduce(&state, .classified(old, .init(status: .ok, method: .model), 1))
        _ = TruckInspectionFeature.reduce(&state, .failed(old, "late"))
        _ = TruckInspectionFeature.reduce(&state, .diagnostic(old, "late"))
        #expect(state == stopping)
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        _ = TruckInspectionFeature.reduce(&state, .start)
        #expect(state.phase == .answering)
        #expect(state.records.isEmpty)
    }

    @Test("TT2: Closing cancels work and reopening waits for the old model to drain")
    func closeAndReopen() throws {
        var state = try state()
        let folder = URL(filePath: "/asset")
        _ = TruckInspectionFeature.reduce(&state, .modelChosen(folder))
        let loadingID = state.generation
        #expect(
            TruckInspectionFeature.reduce(&state, .closed) == [
                .cancelAll, .run(.stop(state.generation, unload: true), id: .drain),
            ])
        let stoppedID = state.generation
        _ = TruckInspectionFeature.reduce(&state, .appeared)
        #expect(state.phase == .stopping)
        #expect(state.reloadAfterStop)
        let info = InspectionModelInfo(folder: folder, revision: "test", milliseconds: 1, computePreference: "gpu")
        _ = TruckInspectionFeature.reduce(&state, .modelLoaded(loadingID, info))
        #expect(state.modelInfo == nil)
        let commands = TruckInspectionFeature.reduce(&state, .stopped(stoppedID, unloaded: true))
        #expect(commands == [.run(.prepare(state.generation, folder), id: .operation)])
        #expect(state.phase == .preparing)
    }

    @Test("TT3: Failed and empty answers remain unanswered and model failures permit retry")
    func failures() throws {
        var state = try state()
        _ = TruckInspectionFeature.reduce(&state, .start)
        #expect(TruckInspectionFeature.reduce(&state, .submit).isEmpty)
        #expect(state.records.isEmpty)
        _ = TruckInspectionFeature.reduce(&state, .draftChanged("Unrecognized answer"))
        #expect(TruckInspectionFeature.reduce(&state, .submit).isEmpty)
        #expect(state.phase == .answering)
        #expect(state.records.isEmpty)
        _ = TruckInspectionFeature.reduce(&state, .stop)
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        let folder = URL(filePath: "/asset")
        _ = TruckInspectionFeature.reduce(&state, .modelChosen(folder))
        _ = TruckInspectionFeature.reduce(&state, .failed(state.generation, "invalid asset"))
        #expect(state.phase == .idle)
        #expect(state.modelInfo == nil)
        #expect(!state.events.isEmpty)
        _ = TruckInspectionFeature.reduce(&state, .modelChosen(folder))
        let info = InspectionModelInfo(folder: folder, revision: "test", milliseconds: 10, computePreference: "gpu")
        _ = TruckInspectionFeature.reduce(&state, .modelLoaded(state.generation, info))
        #expect(state.modelInfo == info)
        _ = TruckInspectionFeature.reduce(&state, .start)
        _ = TruckInspectionFeature.reduce(&state, .submit)
        _ = TruckInspectionFeature.reduce(&state, .failed(state.generation, "classification failed"))
        #expect(state.phase == .answering)
        #expect(state.records.isEmpty)
    }

    @Test("Profile changes reset the record and diagnostics retain at most one thousand events")
    func profileAndLogs() throws {
        var state = try state()
        _ = TruckInspectionFeature.reduce(&state, .start)
        _ = TruckInspectionFeature.reduce(&state, .choose(.ok))
        _ = TruckInspectionFeature.reduce(&state, .stop)
        _ = TruckInspectionFeature.reduce(&state, .stopped(state.generation, unloaded: false))
        _ = TruckInspectionFeature.reduce(&state, .profileChanged(.hydraulic))
        #expect(state.records.isEmpty)
        #expect(state.items.count == 18)
        for index in 0..<1002 {
            _ = TruckInspectionFeature.reduce(&state, .diagnostic(state.generation, "event \(index)"))
        }
        #expect(state.events.count == 1000)
        #expect(state.events.first == "event 2")
        _ = TruckInspectionFeature.reduce(&state, .reset)
        #expect(state.events.isEmpty)
    }
}
