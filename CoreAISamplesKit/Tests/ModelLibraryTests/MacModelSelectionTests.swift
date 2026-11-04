import ModelLibraryFeature
import Testing

struct MacModelSelectionTests {
    @Test("Resetting a pending model choice clears its reservation and rejects its late completion")
    func pendingChoice() {
        var state = MacModelSelection.State()
        #expect(MacModelSelection.reduce(&state, .choose).isEmpty)
        let old = state.request
        #expect(state.selecting && !state.unloading)
        _ = MacModelSelection.reduce(&state, .reset)
        #expect(!state.selecting && !state.unloading && state.request != old)
        _ = MacModelSelection.reduce(&state, .choose)
        let next = state
        _ = MacModelSelection.reduce(&state, .selected(old))
        #expect(state == next)
        _ = MacModelSelection.reduce(&state, .selected(state.request))
        #expect(!state.selecting)
    }

    @Test("An old unload cannot clear a new choice after application termination is cancelled")
    func interruptedUnload() {
        var state = MacModelSelection.State()
        _ = MacModelSelection.reduce(&state, .unload)
        let old = state.request
        #expect(state.unloading && !state.selecting)
        _ = MacModelSelection.reduce(&state, .reset)
        _ = MacModelSelection.reduce(&state, .choose)
        let next = state
        _ = MacModelSelection.reduce(&state, .unloaded(old))
        #expect(state == next && state.selecting)
    }

    @Test("A repeated unload receives a distinct task identity even if the intermediate reset is not rendered")
    func unloadTickets() {
        var state = MacModelSelection.State()
        _ = MacModelSelection.reduce(&state, .unload)
        let old = state.request
        _ = MacModelSelection.reduce(&state, .reset)
        _ = MacModelSelection.reduce(&state, .unload)
        #expect(state.request != old && state.unloading)
        _ = MacModelSelection.reduce(&state, .unloaded(old))
        #expect(state.unloading)
        _ = MacModelSelection.reduce(&state, .unloaded(state.request))
        #expect(!state.unloading)
    }
}
