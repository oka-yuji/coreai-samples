import Foundation
import ReceiptCore

extension ReceiptFeature {
    static func complete(_ state: inout State, operation: ReceiptFinishedOperation, continuing: Bool) -> [Command] {
        let ticket = operation.ticket
        switch (state.phase, operation.completion) {
        case (.samples, .samples(let inputs)):
            state.active = nil
            state.phase = .idle
            if continuing { return reduce(&state, .add(inputs)) }
            _ = append(inputs, to: &state)
        case (.preparing, .prepared):
            state.modelPrepared = true
            if continuing { return next(&state) }
        case (.reading, .read(let read)):
            guard let index = matchingItem(state, ticket), read.record.id == ticket.item,
                read.record.sourceName == state.items[index].input.url.lastPathComponent,
                read.seconds.isFinite, read.seconds >= 0, !read.previews.isEmpty
            else { return invalid(&state) }
            var record = read.record
            if !state.items[index].input.purpose.isEmpty {
                record.edit(.purpose, value: state.items[index].input.purpose)
            }
            state.items[index].record = record
            state.items[index].previews = read.previews
            state.items[index].ocrSeconds = read.seconds
            state.items[index].failure = nil
            if continuing { return next(&state) }
        case (.classifying, .classified(let result)):
            guard let index = matchingItem(state, ticket), state.items[index].record != nil,
                result.seconds.isFinite, result.seconds >= 0, result.tokenCount > 0
            else { return invalid(&state) }
            state.items[index].record?.apply(result.classification)
            state.items[index].needsClassification = false
            state.items[index].decisionSeconds = result.seconds
            state.items[index].tokenCount = result.tokenCount
            state.items[index].failure = nil
            if continuing { return finishItem(&state) }
            state.batchDone += 1
        case (.removing, .removed):
            guard matchingItem(state, ticket) != nil else { return invalid(&state) }
            state.items.removeAll { $0.id == ticket.item }
            if state.selection == ticket.item { state.selection = state.items.first?.id }
            state.saved = nil
            if continuing { return finishItem(&state) }
        case (.exporting, .exported(let saved)):
            guard saved.directory.isFileURL, let ledger = state.exportingLedger,
                saved.count == ledger.rows.count, saved.totalYen == ledger.totalYen
            else { return invalid(&state) }
            state.saved = saved
            if continuing { return finishItem(&state) }
        default: return []
        }
        return []
    }

    private static func matchingItem(_ state: State, _ ticket: ReceiptTicket) -> Int? {
        state.items.firstIndex { $0.id == ticket.item && $0.revision == ticket.revision }
    }

    private static func invalid(_ state: inout State) -> [Command] {
        state.message = "The receipt result did not match the current operation. Try again."
        state.queue = []
        state.exportingLedger = nil
        state.active = nil
        state.phase = .idle
        return []
    }
}
