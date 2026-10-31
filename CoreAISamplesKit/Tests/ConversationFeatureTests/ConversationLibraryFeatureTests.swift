import ConversationFeature
import Foundation
import Testing

struct ConversationLibraryFeatureTests {
    @Test("The active conversation remains reachable before its first save finishes")
    func unsavedCurrentRow() {
        let state = started(.work, id: "unsaved")
        #expect(state.library.items.isEmpty && state.isRunning)
        #expect(state.conversationSummaries.map(\.id) == ["unsaved"])
        #expect(state.conversationSummaries.first?.title == state.title)
    }

    @Test(
        "Quit waits for restoration and uses the restored state on success or the old state on failure",
        arguments: [true, false])
    func quitDuringRestore(succeeds: Bool) {
        var state = started(.chat, id: "old")
        _ = reduce(&state, .generationEnded(run: 1, stopReason: "stopToken"), .archiveSaved(id: "old", revision: 1))
        _ = reduce(&state, .restoreRequested(id: "selected"))
        #expect(reduce(&state, .terminationRequested) == [[]])
        #expect(state.isRestoring && state.isTerminating && state.id == "old")
        let action: ConversationAction =
            succeeds
            ? .archiveRestored(
                id: "selected", .snapshot(ConversationSnapshot(ConversationState(id: "selected", now: launch))))
            : .restoreFailed(id: "selected", message: "Unavailable")
        #expect(reduce(&state, action) == [[.run(.terminationReady(true), id: .termination)]])
        #expect(state.id == (succeeds ? "selected" : "old") && !state.isRestoring)
    }

    @Test("A partial rename adopts the committed revision and retries only its Markdown export")
    func renameFailureState() {
        var state = ConversationState(id: "selected", now: launch)
        state.archiveRevision = 1
        state.persistence = .saved
        state.library.upsert(ConversationSummary(ConversationSnapshot(state)))
        _ = reduce(&state, .library(.renameRequested(id: state.id, title: "Changed", now: launch)))
        var snapshot = ConversationSnapshot(state)
        snapshot.revision = 2
        snapshot.title = "Changed"
        _ = reduce(&state, .library(.renamed(ticket: state.library.ticket, snapshot, warning: "Export failed")))
        #expect(state.title == "Changed" && state.archiveRevision == 2 && state.library.exportRetryID == state.id)
        let expected = state.library.ticket + 1
        #expect(
            reduce(&state, .library(.retryExport)) == [
                [.run(.library(.repairExport(ticket: expected, id: state.id)), id: .libraryMutation)]
            ])
        _ = reduce(&state, .library(.exportRepaired(ticket: expected)))
        #expect(state.library.exportRetryID == nil && state.library.error == nil && state.archiveRevision == 2)
    }

    @Test("Running or saving blocks history changes and a pending mutation blocks a new send")
    func blocksConflictingWork() {
        var state = started(.chat)
        state.library.upsert(
            ConversationSummary(id: state.id, title: "Original", updatedAt: launch, mode: .chat, revision: 1))
        #expect(
            reduce(
                &state, .library(.renameRequested(id: state.id, title: "Renamed", now: launch)),
                .library(.deleteRequested(id: state.id))) == [[], []])
        _ = reduce(&state, .generationEnded(run: 1, stopReason: "stopToken"))
        #expect(!state.canLeave)
        _ = reduce(&state, .archiveSaved(id: state.id, revision: state.archiveRevision))
        _ = reduce(&state, .library(.renameRequested(id: state.id, title: "Renamed", now: launch)))
        #expect(state.library.isMutating && !state.canLeave)
        #expect(reduce(&state, .send("Another", mode: .chat, now: launch), .restoreRequested(id: "other")) == [[], []])
    }

    @Test("Rename synchronizes the selected revision and ignores a list response that started before mutation")
    func renameOrdering() {
        var state = ConversationState(id: "selected", now: launch)
        state.persistence = .saved
        state.archiveRevision = 1
        state.library.upsert(ConversationSummary(ConversationSnapshot(state)))
        _ = reduce(&state, .library(.refresh))
        let listTicket = state.library.ticket
        _ = reduce(&state, .library(.renameRequested(id: state.id, title: "Changed", now: launch)))
        let ticket = state.library.ticket
        var snapshot = ConversationSnapshot(state)
        snapshot.title = "Changed"
        snapshot.revision = 2
        _ = reduce(
            &state, .library(.renamed(ticket: ticket, snapshot, warning: nil)),
            .library(.loaded(ticket: listTicket, [])))
        #expect(state.title == "Changed" && state.archiveRevision == 2 && state.library.items.first?.title == "Changed")
        #expect(!state.library.isMutating && state.canLeave)
    }

    @Test("Deletion needs confirmation and quitting waits until a pending deletion has settled")
    func deletionAndTermination() {
        var state = ConversationState(id: "selected", now: launch)
        state.library.upsert(ConversationSummary(ConversationSnapshot(state)))
        _ = reduce(&state, .library(.deleteRequested(id: state.id)))
        #expect(!state.library.isMutating && state.library.pendingDelete == state.id)
        _ = reduce(&state, .library(.cancelDelete))
        #expect(reduce(&state, .library(.deleteConfirmed(replacementID: "new", now: launch))) == [[]])
        _ = reduce(
            &state, .library(.deleteRequested(id: state.id)),
            .library(.deleteConfirmed(replacementID: "new", now: launch)))
        #expect(reduce(&state, .terminationRequested) == [[]])
        #expect(state.isTerminating && state.library.isMutating)
        #expect(
            reduce(
                &state,
                .library(
                    .deleted(
                        ticket: state.library.ticket, id: state.id, replacementID: "new", now: launch, warning: nil)))
                == [[.run(.terminationReady(true), id: .termination)]])
        #expect(state.id == "new" && state.library.items.isEmpty && state.turns.isEmpty)
    }
}
