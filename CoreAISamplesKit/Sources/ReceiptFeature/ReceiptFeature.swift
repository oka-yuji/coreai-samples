import FeatureStore
import Foundation
import ReceiptCore

public enum ReceiptFeature: Feature {
    public typealias State = ReceiptState
    public typealias Action = ReceiptAction
    public typealias Effect = ReceiptEffect
    public typealias EffectID = ReceiptEffectID
    public typealias Command = FeatureStore.Command<Effect, EffectID>

    public static func reduce(_ state: inout State, _ action: Action) -> [Command] {
        if state.ended {
            if case .released(let ticket) = action, state.active == ticket, state.phase == .closing {
                state.active = nil
                state.phase = .closed
            }
            return []
        }
        if !state.presented {
            switch action {
            case .appeared, .ended: break
            default: return []
            }
        }
        switch action {
        case .appeared:
            state.presented = true
        case .ended:
            state.ended = true
            state.presented = false
            state.queue = []
            state.exportSelection = nil
            state.exportingLedger = nil
            let ticket = begin(&state, phase: .closing)
            return [.cancelAll, .run(.release(ticket), id: .stop)]
        case .add(let inputs):
            guard state.editable, append(inputs, to: &state) else { return [] }
            return process(&state)
        case .samples:
            guard state.editable, state.items.isEmpty else { return [] }
            let ticket = begin(&state, phase: .samples)
            return [.run(.samples(ticket), id: .operation)]
        case .process:
            guard state.canProcess else { return [] }
            return process(&state)
        case .select(let id):
            if state.items.contains(where: { $0.id == id }) { state.selection = id }
        case .edit(let id, let field, let value):
            guard state.editable, let index = state.items.firstIndex(where: { $0.id == id }),
                let record = state.items[index].record, fieldValue(field, in: record.fields) != value
            else { return [] }
            state.items[index].record?.edit(field, value: value)
            state.items[index].revision += 1
            state.items[index].failure = nil
            if field == .merchant || field == .purpose {
                state.items[index].needsClassification = true
                state.items[index].decisionSeconds = nil
                state.items[index].tokenCount = nil
            }
            state.saved = nil
            state.message = nil
        case .category(let id, let category):
            guard state.editable, let index = state.items.firstIndex(where: { $0.id == id }),
                state.items[index].record != nil
            else { return [] }
            state.items[index].record?.setCategory(category)
            state.items[index].revision += 1
            state.items[index].needsClassification = false
            state.items[index].decisionSeconds = nil
            state.items[index].tokenCount = nil
            state.items[index].failure = nil
            state.saved = nil
        case .classify(let id):
            guard state.editable, let index = state.items.firstIndex(where: { $0.id == id }),
                state.items[index].record != nil
            else { return [] }
            let purpose = state.items[index].record?.fields.purpose ?? ""
            state.items[index].record?.edit(.purpose, value: purpose)
            state.items[index].revision += 1
            state.items[index].decisionSeconds = nil
            state.items[index].tokenCount = nil
            state.items[index].needsClassification = true
            state.queue = [id]
            state.batchTotal = 1
            state.batchDone = 0
            state.saved = nil
            return prepareOrNext(&state)
        case .confirm(let id):
            guard state.editable, let index = state.items.firstIndex(where: { $0.id == id }) else { return [] }
            do { try state.items[index].record?.confirm() } catch {
                state.message = "Review the required fields and choose a category before confirming."
            }
        case .confirmTogether:
            guard state.editable else { return [] }
            for index in state.items.indices where state.items[index].record?.canConfirmTogether == true {
                do { try state.items[index].record?.confirm() } catch {
                    state.message = "Some receipts still need individual review."
                }
            }
        case .remove(let id):
            guard state.editable, state.items.contains(where: { $0.id == id }) else { return [] }
            let ticket = begin(&state, phase: .removing, item: id)
            return [.run(.remove(ticket), id: .operation)]
        case .finished(let operation):
            guard state.active == operation.ticket, ![.stopping, .closing, .closed].contains(state.phase) else {
                return []
            }
            return complete(&state, operation: operation, continuing: true)
        case .failed(let ticket, let message):
            guard state.active == ticket, ![.stopping, .closing, .closed].contains(state.phase) else { return [] }
            if let index = state.items.firstIndex(where: { $0.id == ticket.item }) {
                state.items[index].failure = message
            } else {
                state.message = message
            }
            return finishItem(&state)
        case .stop:
            guard state.busy, ![.stopping, .closing, .closed].contains(state.phase) else { return [] }
            state.interrupted = state.active.map { ReceiptInterrupted(ticket: $0, phase: state.phase) }
            state.queue = []
            let previous = state.active
            let ticket = begin(&state, phase: .stopping)
            return [.cancelAll, .run(.stop(ticket, interrupted: previous), id: .stop)]
        case .stopped(let ticket, let operation):
            guard state.active == ticket, state.phase == .stopping else { return [] }
            if let prior = state.interrupted, let operation, operation.ticket == prior.ticket {
                state.active = prior.ticket
                state.phase = prior.phase
                _ = complete(&state, operation: operation, continuing: false)
            }
            state.interrupted = nil
            state.exportingLedger = nil
            state.active = nil
            state.phase = .idle
        case .released: break
        case .requestExport:
            guard state.canExport, let ledger = state.ledger else { return [] }
            state.request += 1
            state.exportSelection = ReceiptExportSelection(
                id: ReceiptTicket(session: state.session, request: state.request), ledger: ledger)
            state.message = nil
        case .exportSelected(let ticket, let directory):
            guard let selection = state.exportSelection, selection.id == ticket else { return [] }
            state.exportSelection = nil
            guard let directory, directory.isFileURL else { return [] }
            state.saved = nil
            state.active = ticket
            state.phase = .exporting
            state.exportingLedger = selection.ledger
            return [.run(.export(ticket, selection.ledger, directory), id: .operation)]
        case .uiError(let message):
            if state.editable { state.message = message }
        }
        return []
    }

    private static func fieldValue(_ field: ReceiptField, in values: ReceiptFields) -> String {
        switch field {
        case .merchant: values.merchant
        case .date: values.date
        case .amount: values.amount
        case .purpose: values.purpose
        }
    }

    static func begin(_ state: inout State, phase: State.Phase, item: Int = 0) -> ReceiptTicket {
        state.request += 1
        let revision = state.items.first(where: { $0.id == item })?.revision ?? 0
        let ticket = ReceiptTicket(session: state.session, request: state.request, item: item, revision: revision)
        state.active = ticket
        state.phase = phase
        state.message = nil
        return ticket
    }

    static func append(_ inputs: [ReceiptInput], to state: inout State) -> Bool {
        guard !inputs.isEmpty else { return false }
        guard inputs.count + state.items.count <= 20 else {
            state.message = "A Receipt Desk session supports up to 20 receipts."
            return false
        }
        guard inputs.allSatisfy({ $0.url.isFileURL }) else {
            state.message = "Choose receipt files from this Mac."
            return false
        }
        for input in inputs {
            state.nextItem += 1
            state.items.append(ReceiptItem(id: state.nextItem, input: input))
        }
        if state.selection == nil { state.selection = state.items.first?.id }
        state.saved = nil
        return true
    }

    private static func process(_ state: inout State) -> [Command] {
        state.queue = state.items.filter { $0.record == nil || $0.needsClassification }.map(\.id)
        state.batchTotal = state.queue.count
        state.batchDone = 0
        guard !state.queue.isEmpty else { return [] }
        return prepareOrNext(&state)
    }

    private static func prepareOrNext(_ state: inout State) -> [Command] {
        if state.modelPrepared { return next(&state) }
        let ticket = begin(&state, phase: .preparing)
        return [.run(.prepare(ticket, state.bundle), id: .operation)]
    }

    static func next(_ state: inout State) -> [Command] {
        guard let id = state.queue.first, let index = state.items.firstIndex(where: { $0.id == id }) else {
            state.active = nil
            state.phase = .idle
            state.queue = []
            return []
        }
        state.items[index].failure = nil
        if let record = state.items[index].record {
            let ticket = begin(&state, phase: .classifying, item: id)
            return [.run(.classify(ticket, record), id: .operation)]
        }
        let ticket = begin(&state, phase: .reading, item: id)
        return [.run(.read(ticket, state.items[index].input.url), id: .operation)]
    }

    static func finishItem(_ state: inout State) -> [Command] {
        if let id = state.active?.item, state.queue.first == id {
            state.queue.removeFirst()
            state.batchDone += 1
            return next(&state)
        }
        state.queue = []
        state.exportingLedger = nil
        state.active = nil
        state.phase = .idle
        return []
    }
}
