import DecideFeature
import FeatureStore
import Foundation
import Testing

struct DecideFeatureTests {
    @Test("Loading a model leads to token counting and stale load results are ignored")
    func loading() {
        var state = readyState()
        state.initialSeconds = 15.2
        state.reuseSeconds = 0.8
        state.output = output()
        state.exportData = Data("previous".utf8)
        #expect(
            DecideFeature.reduce(&state, .openBundle("/bundle")) == [
                .cancel(.count), .run(.load(1, "/bundle"), id: .operation),
            ])
        #expect(state.initialSeconds == nil && state.reuseSeconds == nil)
        #expect(state.output == nil && state.exportData == nil)
        let before = state
        #expect(DecideFeature.reduce(&state, .loaded(0, info)).isEmpty)
        #expect(state == before)
        #expect(
            DecideFeature.reduce(&state, .loaded(1, info)) == [
                .run(.count(1, 0, state.input), id: .count)
            ])
        #expect(state.phase == .idle && state.model == info)
    }

    @Test("Stop cancels the operation, waits for resource completion and ignores late results")
    func stopping() {
        var state = readyState()
        _ = DecideFeature.reduce(&state, .run(reuse: true))
        #expect(
            DecideFeature.reduce(&state, .stop) == [
                .cancel(.operation), .cancel(.count), .run(.waitForIdle(2), id: .stop),
            ])
        let stopping = state
        #expect(DecideFeature.reduce(&state, .completed(1, 0, output())).isEmpty)
        #expect(DecideFeature.reduce(&state, .run(reuse: true)).isEmpty)
        #expect(DecideFeature.reduce(&state, .failed(1, "late")).isEmpty)
        #expect(state == stopping)
        _ = DecideFeature.reduce(&state, .stopped(2))
        #expect(state.phase == .idle)
        #expect(
            DecideFeature.reduce(&state, .run(reuse: true)) == [
                .cancel(.count), .run(.decide(3, 0, state.input, reuse: true), id: .operation),
            ])
    }

    @Test("Changing a checklist clears confirmations and replacement results never inherit them")
    func confirmationsAndChanges() {
        var state = readyState()
        _ = DecideFeature.reduce(&state, .run(reuse: true))
        _ = DecideFeature.reduce(&state, .completed(1, 0, output()))
        _ = DecideFeature.reduce(&state, .confirm(0, "no"))
        #expect(state.confirmations == [0: "no"])
        _ = DecideFeature.reduce(&state, .addItem)
        #expect(state.confirmations.isEmpty && !state.currentResult)
        #expect(state.output != nil)
        _ = DecideFeature.reduce(&state, .run(reuse: true))
        let result = output(probabilities: [0.55, 0.9])
        _ = DecideFeature.reduce(&state, .completed(2, 1, result))
        #expect(state.confirmations.isEmpty && state.currentResult)
        #expect(state.changedItems == [0, 1])
        #expect(state.previous != nil)
    }

    @Test("Cancelled and failed reruns preserve confirmations until a replacement result succeeds")
    func failedRerunsKeepConfirmations() {
        var state = readyState()
        state.output = output()
        state.resultRevision = 0
        _ = DecideFeature.reduce(&state, .confirm(0, "no"))
        _ = DecideFeature.reduce(&state, .run(reuse: true))
        _ = DecideFeature.reduce(&state, .stop)
        _ = DecideFeature.reduce(&state, .stopped(2))
        #expect(state.confirmations == [0: "no"] && state.currentResult)
        _ = DecideFeature.reduce(&state, .run(reuse: true))
        _ = DecideFeature.reduce(&state, .failed(3, "Expected failure."))
        #expect(state.confirmations == [0: "no"] && state.currentResult)
        _ = DecideFeature.reduce(&state, .run(reuse: true))
        _ = DecideFeature.reduce(&state, .completed(4, 0, output()))
        #expect(state.confirmations.isEmpty)
    }

    @Test("Saving an unchanged item or removing a missing item preserves reviewed results")
    func unchangedChecklist() {
        var state = readyState()
        state.output = output()
        state.resultRevision = 0
        _ = DecideFeature.reduce(&state, .confirm(0, "yes"))
        let before = state
        let item = state.input.items[0]
        #expect(DecideFeature.reduce(&state, .updateItem(item)).isEmpty)
        #expect(DecideFeature.reduce(&state, .removeItem(99)).isEmpty)
        #expect(state == before)
    }

    @Test("Rows sort by their largest probability with checklist order as the tie breaker")
    func sorting() {
        var state = readyState()
        state.output = output(probabilities: [0.9, 0.6, 0.6])
        _ = DecideFeature.reduce(&state, .setLowestFirst(true))
        #expect(state.rows.map(\.id) == [1, 2, 0])
        _ = DecideFeature.reduce(&state, .setLowestFirst(false))
        #expect(state.rows.map(\.id) == [0, 1, 2])
    }

    @Test("Document edits invalidate previous results and raw Unicode edits are preserved")
    func documentEdits() {
        var state = readyState()
        state.input.document = "\u{00e9}"
        state.output = output()
        state.resultRevision = 0
        _ = DecideFeature.reduce(&state, .setDocument("e\u{0301}"))
        #expect(state.revision == 1)
        #expect(state.input.document.utf8.elementsEqual("e\u{0301}".utf8))
        #expect(state.output == nil && state.previous == nil)
    }

    @Test("Ending the feature cancels every effect and rejects input until it appears again")
    func ending() {
        var state = readyState()
        #expect(DecideFeature.reduce(&state, .ended) == [.cancelAll])
        let closed = state
        #expect(DecideFeature.reduce(&state, .setDocument("new")).isEmpty)
        #expect(DecideFeature.reduce(&state, .completed(0, 0, output())).isEmpty)
        #expect(state == closed)
        _ = DecideFeature.reduce(&state, .appeared)
        #expect(state.phase == .idle)
    }

    @Test("Failures and stale token counts cannot overwrite newer state")
    func errorsAndCounts() {
        var state = readyState()
        _ = DecideFeature.reduce(&state, .run(reuse: false))
        _ = DecideFeature.reduce(&state, .failed(1, "Could not evaluate."))
        #expect(state.phase == .idle && state.message == "Could not evaluate.")
        _ = DecideFeature.reduce(&state, .setDocument("another"))
        _ = DecideFeature.reduce(&state, .counted(1, 0, 10))
        #expect(state.inputTokens == nil)
        _ = DecideFeature.reduce(&state, .countFailed(1, 1, "Too many tokens."))
        #expect(!state.canRun)
        _ = DecideFeature.reduce(&state, .counted(1, 1, 20))
        #expect(state.canRun && state.inputTokens == 20)
        _ = DecideFeature.reduce(&state, .failed(1, "late"))
        #expect(state.message == nil)
    }

    @Test("Delayed export preparation, completion and cancellation cannot replace a newer export")
    func delayedExport() throws {
        var state = readyState()
        state.output = output()
        state.resultRevision = 0
        _ = DecideFeature.reduce(&state, .confirm(0, "yes"))
        _ = DecideFeature.reduce(&state, .prepareExport)
        let previous = try #require(state.exportTicket)
        _ = DecideFeature.reduce(&state, .confirm(0, "no"))
        _ = DecideFeature.reduce(&state, .exportPrepared(previous, Data("stale".utf8)))
        #expect(state.exportData == nil)
        _ = DecideFeature.reduce(&state, .ended)
        _ = DecideFeature.reduce(&state, .appeared)
        _ = DecideFeature.reduce(&state, .prepareExport)
        let current = try #require(state.exportTicket)
        #expect(current.id != previous.id)
        _ = DecideFeature.reduce(&state, .exportPrepared(current, Data("current".utf8)))
        _ = DecideFeature.reduce(&state, .exportFinished(previous, nil))
        _ = DecideFeature.reduce(&state, .dismissExport(previous))
        #expect(state.exportData == Data("current".utf8))
        _ = DecideFeature.reduce(&state, .exportFinished(current, nil))
        #expect(state.exportData == nil && state.message == "JSON exported.")
    }

    @Test("Dismissed export preparation cannot reopen a dialog and repeated exports get distinct IDs")
    func dismissedExport() throws {
        var state = readyState()
        state.output = output()
        state.resultRevision = 0
        _ = DecideFeature.reduce(&state, .prepareExport)
        let first = try #require(state.exportTicket)
        _ = DecideFeature.reduce(&state, .dismissExport(first))
        _ = DecideFeature.reduce(&state, .exportPrepared(first, Data("late".utf8)))
        #expect(state.exportData == nil)
        _ = DecideFeature.reduce(&state, .prepareExport)
        let second = try #require(state.exportTicket)
        #expect(second.id != first.id)
        #expect(second.session == first.session && second.revision == first.revision)
    }

    @Test("A delayed save completion cannot overwrite a later import failure")
    func exportAfterImport() throws {
        var state = readyState()
        state.output = output()
        state.resultRevision = 0
        _ = DecideFeature.reduce(&state, .prepareExport)
        let ticket = try #require(state.exportTicket)
        _ = DecideFeature.reduce(&state, .exportPrepared(ticket, Data("export".utf8)))
        _ = DecideFeature.reduce(&state, .dismissExport(ticket))
        _ = DecideFeature.reduce(&state, .readDocument(URL(filePath: "/document.txt")))
        _ = DecideFeature.reduce(&state, .failed(1, "Import failed."))
        _ = DecideFeature.reduce(&state, .exportFinished(ticket, nil))
        _ = DecideFeature.reduce(&state, .exportFailed(ticket, "Late export error."))
        _ = DecideFeature.reduce(&state, .dismissExport(ticket))
        #expect(state.message == "Import failed.")
    }

    @Test("Bundled examples load independently of an ongoing model load and never overwrite pasted text")
    func examples() {
        var state = DecideState()
        _ = DecideFeature.reduce(&state, .appeared)
        let exampleSession = state.exampleSession
        _ = DecideFeature.reduce(&state, .setDocument("my text"))
        _ = DecideFeature.reduce(&state, .openBundle("/bundle"))
        let example = DecideExample(
            id: "example", title: "Example", input: input, extraItem: .blank(id: 1), searchHint: "text")
        _ = DecideFeature.reduce(&state, .examplesLoaded(exampleSession, [example]))
        #expect(state.examples == [example] && !state.loadingExamples)
        #expect(state.input.document == "my text")
        #expect(state.phase == .loading)
    }

    @Test("JSON keeps the model candidate and the reviewer confirmation in separate fields")
    func export() throws {
        let result = output()
        let data = try DecideExport.data(output: result, confirmations: [0: "no"])
        let root = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let confirmations = try #require(root["reviewer_confirmations"] as? [[String: String]])
        #expect(confirmations == [["question": "q0", "option": "no", "status": "Confirmed"]])
        let response = try #require(root["response"] as? [String: Any])
        let answers = try #require(response["answers"] as? [String: [String: String]])
        #expect(answers["q0"]?["choice"] == "yes")
        #expect(throws: DecideExport.ExportError.invalidConfirmation) {
            try DecideExport.data(output: result, confirmations: [0: "absent"])
        }
    }

    @Test("Checklist validation rejects duplicate or unsupported items without collapsing scalar-distinct IDs")
    func validation() {
        var value = input
        value.items.append(value.items[0])
        #expect(value.problem != nil)
        value = input
        value.items = (0..<17).map { .blank(id: $0) }
        #expect(value.problem != nil)
        value.items = [.blank(id: 0), .blank(id: 1)]
        value.items[0].key = "\u{00e9}"
        value.items[1].key = "e\u{0301}"
        #expect(value.problem == nil)
    }

    private var input: DecideInput { DecideInput(document: "Document", items: [.blank(id: 0)]) }
    private var info: DecideBundleInfo { DecideBundleInfo(name: "fixture", path: "/bundle", tokenLimit: 9_216) }

    private func readyState() -> DecideState {
        var state = DecideState(input: input)
        state.model = info
        return state
    }

    private func output(probabilities: [Float] = [0.9]) -> DecideOutput {
        let rows = probabilities.enumerated().map { index, probability in
            DecideRow(
                id: index, key: "q\(index)", candidate: "yes",
                probabilities: [
                    DecideProbability(id: 0, option: "yes", label: "Yes", probability: probability),
                    DecideProbability(id: 1, option: "no", label: "No", probability: 1 - probability),
                ], reviewFirst: probability < 0.6)
        }
        return DecideOutput(
            rows: rows, modelName: "fixture", modelVersion: "v1", inputTokens: 20, reusedTokens: 10,
            seconds: 0.5, reusedDocument: true, recomputedAll: true,
            requestJSON: #"{"state":"Document","questions":{}}"#,
            responseJSON: #"{"answers":{"q0":{"choice":"yes"}}}"#)
    }
}
