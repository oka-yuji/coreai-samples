import ConversationFeature
import Foundation
import Testing

struct ConversationPersistenceTests {
    @Test("Only the current save result unlocks switching and a failed save can be retried")
    func saveOrdering() {
        var state = started(.work)
        _ = reduce(&state, .stepFinished(run: 1, stats(1)), .generationEnded(run: 1, stopReason: "stopToken"))
        #expect(state.archiveRevision == 2 && !state.canLeave)
        let pending = state
        _ = reduce(
            &state, .archiveSaved(id: state.id, revision: 1),
            .archiveFailed(id: "another", revision: 2, message: "failure"))
        #expect(state == pending)
        _ = reduce(&state, .archiveFailed(id: state.id, revision: 2, message: "Disk full"))
        #expect(!state.canLeave && state.persistence == .failed("Disk full"))
        _ = reduce(&state, .retrySave)
        #expect(state.archiveRevision == 3 && state.isSaving)
        _ = reduce(&state, .archiveSaved(id: state.id, revision: 3))
        #expect(state.canLeave && state.persistence == .saved)
    }

    @Test("Quitting waits for stop and the final save instead of accepting an earlier checkpoint")
    func termination() {
        var state = started(.work)
        _ = reduce(&state, .stepFinished(run: 1, stats(1)))
        #expect(reduce(&state, .terminationRequested) == [[.run(.stop(run: 1), id: .stop)]])
        #expect(state.isTerminating && state.stopRequested)
        #expect(reduce(&state, .archiveSaved(id: state.id, revision: 1)) == [[]])
        _ = reduce(&state, .stopped(run: 1))
        #expect(!state.isRunning && state.isSaving)
        #expect(reduce(&state, .archiveSaved(id: state.id, revision: 2)) == [[.run(.terminationReady(true), id: nil)]])
        #expect(!state.canLeave)
        #expect(reduce(&state, .sendRepair("Repair citations", now: launch)) == [[.run(.cancelRepair, id: nil)]])
        _ = reduce(&state, .terminationCancelled)
        #expect(state.canLeave)
    }

    @Test("Stopping preparation keeps the stopping state until acknowledgement and ignores later progress")
    func stopsPreparation() {
        var state = started(.chat)
        _ = reduce(&state, .preparationChanged(run: 1, processed: 256, total: 2_048))
        #expect(state.preparation == ConversationPreparation(processed: 256, total: 2_048))
        _ = reduce(&state, .stopRequested, .preparationChanged(run: 1, processed: 2_048, total: 2_048))
        #expect(state.status == .stopping && state.isRunning)
        _ = reduce(&state, .stopped(run: 1))
        #expect(state.preparation == nil && !state.isRunning && state.isSaving)
    }

    @Test("Startup opens history without a model and unsupported history cannot be sent or overwritten")
    func startup() {
        var state = ConversationState(id: "new", now: launch)
        #expect(reduce(&state, .restoreLatestRequested) == [[.run(.restoreLatest, id: .restore)]])
        #expect(!state.canLeave)
        _ = reduce(&state, .latestRestored(.readOnly(id: "old", markdown: "Old notes", reason: "New format")))
        #expect(state.id == "old" && state.readOnlyMarkdown == "Old notes" && state.canLeave)
        #expect(reduce(&state, .send("Continue", mode: .work, now: launch), .retrySave) == [[], []])
    }
}
