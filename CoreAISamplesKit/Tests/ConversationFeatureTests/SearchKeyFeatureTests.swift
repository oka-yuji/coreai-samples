import ConversationFeature
import FeatureStore
import Testing

struct SearchKeyFeatureTests {
    typealias KeyFeature = SearchKeyFeature

    @Test("Loading starts once, and a result from an earlier ticket is ignored")
    func loadOnceAndIgnoreStaleResult() {
        var state = KeyFeature.State()
        #expect(KeyFeature.reduce(&state, .loadRequested) == [.run(.load(ticket: 1), id: .keychain)])
        #expect(KeyFeature.reduce(&state, .loadRequested).isEmpty)
        _ = KeyFeature.reduce(&state, .replaced("headless-key"))
        _ = KeyFeature.reduce(&state, .loaded(ticket: 1, .stored("late-key")))
        #expect(state.saved == "headless-key" && state.draft == "headless-key" && state.phase == .ready)
    }

    @Test(
        "A failed save keeps the saved key and the draft and reports the failure with a retry",
        arguments: [("new-key", KeyFeature.Failure.Operation.write), ("", .delete)])
    func failedSaveKeepsSavedKey(draft: String, operation: KeyFeature.Failure.Operation) {
        var state = loaded("old-key")
        _ = KeyFeature.reduce(&state, .draftChanged(draft))
        let failure = KeyFeature.Failure(operation, reason: "Denied")
        #expect(KeyFeature.reduce(&state, .saveRequested) == [.run(.save(draft, ticket: 2), id: .keychain)])
        #expect(state.status == .saving && !state.canEdit)
        _ = KeyFeature.reduce(&state, .saved(ticket: 2, .failure(failure)))
        #expect(state.saved == "old-key" && state.draft == draft)
        #expect(state.status == .failed(failure) && state.canSave)
        #expect(KeyFeature.reduce(&state, .saveRequested) == [.run(.save(draft, ticket: 3), id: .keychain)])
    }

    @Test("A read failure is shown as unreadable instead of an empty key, and retrying reads again")
    func readFailureIsDistinct() {
        var state = KeyFeature.State()
        _ = KeyFeature.reduce(&state, .loadRequested)
        let failure = KeyFeature.Failure(.read, reason: "Locked")
        _ = KeyFeature.reduce(&state, .loaded(ticket: 1, .unreadable(failure)))
        #expect(state.status == .failed(failure) && state.status != .empty)
        #expect(state.saved.isEmpty && !state.canSave)
        #expect(KeyFeature.reduce(&state, .loadRequested) == [.run(.load(ticket: 2), id: .keychain)])
        _ = KeyFeature.reduce(&state, .loaded(ticket: 2, .stored("")))
        #expect(state.status == .empty)
    }

    @Test("A legacy key that could not be moved stays in use and can be saved again from the same draft")
    func unmovedKeyCanBeSaved() {
        var state = KeyFeature.State()
        _ = KeyFeature.reduce(&state, .loadRequested)
        let failure = KeyFeature.Failure(.move, reason: "Denied")
        _ = KeyFeature.reduce(&state, .loaded(ticket: 1, .unmoved("legacy-key", failure)))
        #expect(state.saved == "legacy-key" && state.draft == "legacy-key" && state.status == .failed(failure))
        #expect(KeyFeature.reduce(&state, .saveRequested) == [.run(.save("legacy-key", ticket: 2), id: .keychain)])
        _ = KeyFeature.reduce(&state, .saved(ticket: 2, .success("legacy-key")))
        #expect(state.status == .saved && state.failure == nil)
    }

    @Test("A legacy key whose move fails again on retry can be retried again with the same draft")
    func unmovedKeyRetriesAfterRepeatedFailures() {
        var state = KeyFeature.State()
        _ = KeyFeature.reduce(&state, .loadRequested)
        let failure = KeyFeature.Failure(.move, reason: "Denied")
        _ = KeyFeature.reduce(&state, .loaded(ticket: 1, .unmoved("legacy-key", failure)))
        _ = KeyFeature.reduce(&state, .saveRequested)
        _ = KeyFeature.reduce(&state, .saved(ticket: 2, .failure(KeyFeature.Failure(.write, reason: "Denied again"))))
        #expect(state.saved == "legacy-key" && state.draft == "legacy-key")
        #expect(state.status == .failed(KeyFeature.Failure(.move, reason: "Denied again")))
        #expect(state.canSave && state.retry == .saveRequested)
        #expect(KeyFeature.reduce(&state, .saveRequested) == [.run(.save("legacy-key", ticket: 3), id: .keychain)])
        _ = KeyFeature.reduce(&state, .saved(ticket: 3, .success("legacy-key")))
        #expect(state.status == .saved && !state.canSave)
    }

    @Test("After a read failure, a failed save of a new key keeps the draft and the saved key can still be read again")
    func readFailureSurvivesFailedSave() {
        var state = KeyFeature.State()
        _ = KeyFeature.reduce(&state, .loadRequested)
        _ = KeyFeature.reduce(&state, .loaded(ticket: 1, .unreadable(KeyFeature.Failure(.read, reason: "Locked"))))
        _ = KeyFeature.reduce(&state, .draftChanged("new-key"))
        _ = KeyFeature.reduce(&state, .saveRequested)
        let failure = KeyFeature.Failure(.write, reason: "Locked")
        _ = KeyFeature.reduce(&state, .saved(ticket: 2, .failure(failure)))
        #expect(state.saved.isEmpty && state.draft == "new-key" && state.status == .failed(failure) && state.canSave)
        #expect(KeyFeature.reduce(&state, .loadRequested) == [.run(.load(ticket: 3), id: .keychain)])
        _ = KeyFeature.reduce(&state, .loaded(ticket: 3, .stored("stored-key")))
        #expect(state.saved == "stored-key" && state.draft == "stored-key" && state.status == .saved)
    }

    @Test("Retry repeats the failed operation, and reads the saved key again once there is nothing left to save")
    func retryRepeatsTheFailedOperation() {
        var state = KeyFeature.State()
        _ = KeyFeature.reduce(&state, .loadRequested)
        _ = KeyFeature.reduce(&state, .loaded(ticket: 1, .unreadable(KeyFeature.Failure(.read, reason: "Locked"))))
        _ = KeyFeature.reduce(&state, .draftChanged("new-key"))
        #expect(state.retry == .loadRequested)
        _ = KeyFeature.reduce(&state, .saveRequested)
        #expect(state.retry == nil)
        _ = KeyFeature.reduce(&state, .saved(ticket: 2, .failure(KeyFeature.Failure(.write, reason: "Locked"))))
        #expect(state.retry == .saveRequested)
        _ = KeyFeature.reduce(&state, .draftChanged(""))
        #expect(state.retry == .loadRequested)
        state = loaded("old-key")
        _ = KeyFeature.reduce(&state, .draftChanged("new-key"))
        _ = KeyFeature.reduce(&state, .saveRequested)
        _ = KeyFeature.reduce(&state, .saved(ticket: 2, .failure(KeyFeature.Failure(.write, reason: "Denied"))))
        _ = KeyFeature.reduce(&state, .draftChanged("old-key"))
        #expect(state.retry == nil)
    }

    @Test("Closing waits for a key operation and is refused while a failed save or delete can be retried")
    func closingWaitsForTheKey() {
        var state = KeyFeature.State()
        #expect(state.canLeave)
        _ = KeyFeature.reduce(&state, .loadRequested)
        #expect(!state.canLeave)
        _ = KeyFeature.reduce(&state, .loaded(ticket: 1, .stored("old-key")))
        _ = KeyFeature.reduce(&state, .draftChanged(""))
        #expect(state.canLeave)
        _ = KeyFeature.reduce(&state, .saveRequested)
        #expect(!state.canLeave)
        _ = KeyFeature.reduce(&state, .saved(ticket: 2, .failure(KeyFeature.Failure(.delete, reason: "Denied"))))
        #expect(!state.canLeave)
        _ = KeyFeature.reduce(&state, .draftChanged("old-key"))
        #expect(state.canLeave)
    }

    @Test(
        "After loading, closing is refused only while the legacy key has not been moved",
        arguments: [
            (KeyFeature.Load.stored("key"), true),
            (.unmoved("legacy-key", KeyFeature.Failure(.move, reason: "Denied")), false),
            (.unreadable(KeyFeature.Failure(.read, reason: "Locked")), true),
        ])
    func closingAfterLoading(load: KeyFeature.Load, canLeave: Bool) {
        var state = KeyFeature.State()
        _ = KeyFeature.reduce(&state, .loadRequested)
        _ = KeyFeature.reduce(&state, .loaded(ticket: 1, load))
        #expect(state.canLeave == canLeave)
    }

    @Test("The status follows reading, editing and saving the key")
    func statusFollowsTheKey() {
        var state = KeyFeature.State()
        #expect(state.status == .reading && !state.canEdit)
        state = loaded("")
        #expect(state.status == .empty && !state.canSave)
        _ = KeyFeature.reduce(&state, .draftChanged("new-key"))
        #expect(state.status == .unsaved && state.canSave)
        _ = KeyFeature.reduce(&state, .saveRequested)
        #expect(KeyFeature.reduce(&state, .saveRequested).isEmpty)
        _ = KeyFeature.reduce(&state, .saved(ticket: 2, .success("new-key")))
        #expect(state.status == .saved && state.saved == "new-key")
    }

    private func loaded(_ key: String) -> KeyFeature.State {
        var state = KeyFeature.State()
        _ = KeyFeature.reduce(&state, .loadRequested)
        _ = KeyFeature.reduce(&state, .loaded(ticket: 1, .stored(key)))
        return state
    }
}
