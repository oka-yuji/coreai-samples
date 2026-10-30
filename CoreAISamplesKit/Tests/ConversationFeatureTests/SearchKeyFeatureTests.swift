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
