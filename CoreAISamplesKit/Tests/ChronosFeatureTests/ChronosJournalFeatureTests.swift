import ChronosFeature
import FeatureStore
import Foundation
import Testing

struct ChronosJournalFeatureTests {
    @Test("J1 Only native successes enqueue saves and retries retain the captured batch")
    func captures() {
        var state = ChronosJournalFeature.State()
        let preview = batch(.precomputed)
        #expect(ChronosJournalFeature.reduce(&state, .capture(preview)).isEmpty)
        let native = batch(.live)
        #expect(ChronosJournalFeature.reduce(&state, .capture(native)) == [.run(.save(1, native), id: .save)])
        #expect(ChronosJournalFeature.reduce(&state, .capture(native)).isEmpty)
        _ = ChronosJournalFeature.reduce(&state, .saveFailed(1, .persistence))
        #expect(state.saves.count == 2 && state.saveError == .persistence)
        #expect(ChronosJournalFeature.reduce(&state, .retrySave) == [.run(.save(1, native), id: .save)])
        #expect(ChronosJournalFeature.reduce(&state, .saved(1, UUID())) == [.run(.save(2, native), id: .save)])
        _ = ChronosJournalFeature.reduce(&state, .saved(2, UUID()))
        #expect(state.saves.isEmpty && !state.saving && state.saveError == nil)
    }

    @Test("J2 Closing invalidates checks and ended sessions reject later callbacks and captures")
    func lifecycle() {
        var state = ChronosJournalFeature.State()
        let id = UUID()
        #expect(ChronosJournalFeature.reduce(&state, .open) == [.run(.list(1), id: .operation)])
        #expect(ChronosJournalFeature.reduce(&state, .select(id)) == [.run(.detail(2, id), id: .operation)])
        #expect(ChronosJournalFeature.reduce(&state, .close) == [.cancel(.operation)])
        _ = ChronosJournalFeature.reduce(&state, .failed(2, .network))
        #expect(state.error == nil && !state.busy)
        #expect(ChronosJournalFeature.reduce(&state, .ended) == [.cancelAll])
        let ended = state
        _ = ChronosJournalFeature.reduce(&state, .appear)
        _ = ChronosJournalFeature.reduce(&state, .capture(batch(.live)))
        #expect(state == ended)
    }

    private func batch(_ mode: ChronosForecastMode) -> ChronosForecastBatch {
        ChronosForecastBatch(scenarioID: "fixed", originStep: 0, mode: mode, areas: [], processingSeconds: nil)
    }
}
