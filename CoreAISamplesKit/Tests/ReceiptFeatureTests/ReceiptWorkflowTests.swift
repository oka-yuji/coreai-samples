import FeatureStore
import Foundation
import ReceiptCore
import Testing

@testable import ReceiptFeature

struct ReceiptWorkflowTests {
    @Test("Receipts advance one at a time and stale completions cannot alter another item")
    func sequence() throws {
        var state = try readingState(count: 2)
        let first = try #require(state.active)
        _ = finish(&state, .read(read(1)))
        #expect(state.phase == .classifying && state.readCount == 1)
        let firstDecision = try #require(state.active)
        _ = finish(&state, .classified(try decision()))
        #expect(state.phase == .reading && state.active?.item == 2)
        let second = state
        _ = ReceiptFeature.reduce(
            &state, .finished(ReceiptFinishedOperation(ticket: first, completion: .read(read(1)))))
        _ = ReceiptFeature.reduce(
            &state, .finished(ReceiptFinishedOperation(ticket: firstDecision, completion: .classified(try decision()))))
        #expect(state == second)
        _ = finish(&state, .read(read(2)))
        _ = finish(&state, .classified(try decision()))
        #expect(state.phase == .idle && state.readCount == 2 && state.batchDone == 2)
        #expect(state.confirmedCount == 0 && state.ledger == nil)
    }

    @Test("Editing invalidates confirmation and a purpose edit requires a fresh category")
    func edits() throws {
        var state = try completedState()
        _ = ReceiptFeature.reduce(&state, .confirm(1))
        #expect(state.ledger?.totalYen == 1000)
        _ = ReceiptFeature.reduce(&state, .edit(1, .amount, "2500"))
        #expect(state.confirmedCount == 0 && state.ledger == nil)
        #expect(state.items[0].record?.fields.category == .supplies)
        _ = ReceiptFeature.reduce(&state, .confirm(1))
        #expect(state.ledger?.totalYen == 2500)
        _ = ReceiptFeature.reduce(&state, .edit(1, .purpose, "A personal gift"))
        #expect(state.items[0].record?.fields.category == .review)
        #expect(state.items[0].needsClassification && state.items[0].decisionSeconds == nil)
        _ = ReceiptFeature.reduce(&state, .confirm(1))
        #expect(state.confirmedCount == 0)
        _ = ReceiptFeature.reduce(&state, .category(1, .other))
        _ = ReceiptFeature.reduce(&state, .confirm(1))
        #expect(state.ledger?.totalYen == 2500)
        _ = ReceiptFeature.reduce(&state, .classify(1))
        #expect(state.items[0].record?.fields.category == .review && state.confirmedCount == 0)
        let ticket = try #require(state.active)
        _ = ReceiptFeature.reduce(&state, .failed(ticket, "Classification failed"))
        #expect(state.phase == .idle && state.items[0].record?.canConfirm == false)
    }

    @Test("Unchanged field binding notifications preserve the reviewed category and confirmation")
    func unchangedBindings() throws {
        var state = try completedState()
        _ = ReceiptFeature.reduce(&state, .confirm(1))
        let before = state
        for (field, value) in [
            (ReceiptField.merchant, "Shop"), (.date, "2026-11-06"), (.amount, "1000"), (.purpose, ""),
        ] {
            #expect(ReceiptFeature.reduce(&state, .edit(1, field, value)).isEmpty)
            #expect(state == before)
        }
    }

    @Test("Stop joins the operation and preserves a read that committed before cancellation")
    func reconcileRead() throws {
        var state = try readingState()
        let reading = try #require(state.active)
        let operation = ReceiptFinishedOperation(ticket: reading, completion: .read(read(1)))
        let commands = ReceiptFeature.reduce(&state, .stop)
        let stopping = try #require(state.active)
        #expect(commands == [.cancelAll, .run(.stop(stopping, interrupted: reading), id: .stop)])
        let stoppedState = state
        #expect(ReceiptFeature.reduce(&state, .process).isEmpty)
        #expect(ReceiptFeature.reduce(&state, .finished(operation)).isEmpty)
        #expect(state == stoppedState)
        _ = ReceiptFeature.reduce(&state, .stopped(stopping, operation))
        #expect(state.phase == .idle && state.readCount == 1)
        let next = ReceiptFeature.reduce(&state, .process)
        #expect(state.phase == .classifying)
        #expect(next.count == 1)
    }

    @Test("A classification recovered during stop advances progress without starting the next receipt")
    func stoppedClassificationProgress() throws {
        var state = try readingState(count: 2)
        _ = finish(&state, .read(read(1)))
        let classified = try #require(state.active)
        _ = ReceiptFeature.reduce(&state, .stop)
        let stopped = try #require(state.active)
        _ = ReceiptFeature.reduce(
            &state,
            .stopped(stopped, ReceiptFinishedOperation(ticket: classified, completion: .classified(try decision()))))
        #expect(state.phase == .idle && state.batchDone == 1 && state.batchTotal == 2)
        #expect(state.items[0].needsClassification == false && state.items[1].record == nil)
    }

    @Test("A stopped export acknowledges publication and the picker freezes the confirmed snapshot")
    func exportCommit() throws {
        var state = try completedState()
        _ = ReceiptFeature.reduce(&state, .confirm(1))
        _ = ReceiptFeature.reduce(&state, .requestExport)
        let panel = try #require(state.exportSelection)
        let frozen = state
        _ = ReceiptFeature.reduce(&state, .edit(1, .amount, "9999"))
        #expect(state == frozen)
        let folder = URL(filePath: "/chosen")
        let commands = ReceiptFeature.reduce(&state, .exportSelected(panel.id, folder))
        #expect(commands == [.run(.export(panel.id, panel.ledger, folder), id: .operation)])
        _ = ReceiptFeature.reduce(&state, .stop)
        let stopping = try #require(state.active)
        let saved = ReceiptSaved(directory: folder.appending(path: "ReceiptDesk-example"), count: 1, totalYen: 1000)
        _ = ReceiptFeature.reduce(
            &state, .stopped(stopping, ReceiptFinishedOperation(ticket: panel.id, completion: .exported(saved))))
        #expect(state.phase == .idle && state.saved == saved)
        #expect(state.ledger?.totalYen == 1000)
        _ = ReceiptFeature.reduce(&state, .exportSelected(panel.id, folder))
        #expect(state.phase == .idle)
    }

    @Test("A failed receipt stays visible while the next receipt proceeds and limits do not silently truncate")
    func failures() throws {
        var state = try readingState(count: 2)
        let first = try #require(state.active)
        _ = ReceiptFeature.reduce(&state, .failed(first, "Unreadable file"))
        #expect(state.items[0].failure == "Unreadable file")
        #expect(state.phase == .reading && state.active?.item == 2)
        let current = state
        _ = ReceiptFeature.reduce(&state, .failed(first, "Late failure"))
        #expect(state == current)
        var empty = ReceiptState(bundle: URL(filePath: "/model"))
        _ = ReceiptFeature.reduce(&empty, .appeared)
        #expect(
            ReceiptFeature.reduce(&empty, .add((1...21).map { ReceiptInput(url: URL(filePath: "/\($0).png")) })).isEmpty
        )
        #expect(empty.items.isEmpty && empty.message != nil)
    }

    @Test("Bulk confirmation leaves warnings for individual review and overflow is visible")
    func totals() throws {
        var state = try readingState(count: 2)
        _ = finish(&state, .read(read(1, amount: String(Int64.max))))
        _ = finish(&state, .classified(try decision()))
        _ = finish(&state, .read(read(2, amount: "1", issues: [.lowRecognitionConfidence])))
        _ = finish(&state, .classified(try decision()))
        _ = ReceiptFeature.reduce(&state, .confirmTogether)
        #expect(state.confirmedCount == 1 && state.ledger?.totalYen == Int64.max)
        _ = ReceiptFeature.reduce(&state, .confirm(2))
        #expect(state.confirmedCount == 2 && state.ledger == nil && state.ledgerProblem != nil)
        #expect(!state.canExport)
    }

    @Test("Cancelling the folder chooser unlocks the same reviewed draft and allows another export")
    func cancelledPicker() throws {
        var state = try completedState()
        _ = ReceiptFeature.reduce(&state, .confirm(1))
        _ = ReceiptFeature.reduce(&state, .requestExport)
        let first = try #require(state.exportSelection)
        _ = ReceiptFeature.reduce(&state, .exportSelected(first.id, nil))
        #expect(state.editable && state.ledger?.totalYen == 1000 && state.items[0].isConfirmed)
        _ = ReceiptFeature.reduce(&state, .requestExport)
        let second = try #require(state.exportSelection)
        #expect(second.id != first.id)
        _ = ReceiptFeature.reduce(&state, .exportSelected(first.id, nil))
        #expect(state.exportSelection == second)
        _ = ReceiptFeature.reduce(&state, .exportSelected(second.id, nil))
        #expect(state.editable)
    }

    @Test("Wrong sessions and revisions cannot complete classification")
    func identity() throws {
        var state = try readingState()
        _ = finish(&state, .read(read(1)))
        let ticket = try #require(state.active)
        let before = state
        for wrong in [
            ReceiptTicket(
                session: ticket.session + 1, request: ticket.request, item: ticket.item, revision: ticket.revision),
            ReceiptTicket(
                session: ticket.session, request: ticket.request, item: ticket.item, revision: ticket.revision + 1),
        ] {
            _ = ReceiptFeature.reduce(
                &state, .finished(ReceiptFinishedOperation(ticket: wrong, completion: .classified(try decision()))))
        }
        #expect(state == before)
    }

    private func readingState(count: Int = 1) throws -> ReceiptState {
        var state = ReceiptState(bundle: URL(filePath: "/model"), session: 11)
        _ = ReceiptFeature.reduce(&state, .appeared)
        _ = ReceiptFeature.reduce(
            &state, .add((1...count).map { ReceiptInput(url: URL(filePath: "/receipt\($0).png")) }))
        _ = finish(&state, .prepared)
        #expect(state.phase == .reading)
        return state
    }

    private func completedState() throws -> ReceiptState {
        var state = try readingState()
        _ = finish(&state, .read(read(1)))
        _ = finish(&state, .classified(try decision()))
        return state
    }

    private func finish(_ state: inout ReceiptState, _ completion: ReceiptCompletion) -> [ReceiptFeature.Command] {
        guard let ticket = state.active else {
            Issue.record("An active operation is required")
            return []
        }
        return ReceiptFeature.reduce(
            &state, .finished(ReceiptFinishedOperation(ticket: ticket, completion: completion)))
    }

    private func read(_ id: Int, amount: String = "1000", issues: [ReceiptIssue] = []) -> ReceiptRead {
        ReceiptRead(
            record: ReceiptRecord(
                id: id, sourceName: "receipt\(id).png", sourceExtension: "png",
                sourceSHA256: String(repeating: "a", count: 64),
                text: "Shop\n2026-11-06\nTotal JPY " + amount,
                extraction: ReceiptExtraction(
                    fields: ReceiptFields(merchant: "Shop", date: "2026-11-06", amount: amount), issues: issues)),
            previews: [Data([1])], seconds: 0.2)
    }

    private func decision() throws -> ReceiptDecision {
        ReceiptDecision(
            classification: try ReceiptClassification(probabilities: [0, 0, 1, 0, 0, 0, 0, 0]), tokenCount: 100,
            seconds: 0.3)
    }
}
