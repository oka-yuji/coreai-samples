import FeatureStore
import Foundation
import ReceiptCore
import Testing

@testable import ReceiptFeature

struct ReceiptFeatureTests {
    @Test("Adding receipts starts preparation only after an explicit action")
    func start() throws {
        var state = ReceiptState(bundle: URL(filePath: "/model"))
        #expect(ReceiptFeature.reduce(&state, .appeared).isEmpty)
        #expect(state.phase == .idle && !state.modelPrepared)
        let commands = ReceiptFeature.reduce(&state, .add([ReceiptInput(url: URL(filePath: "/receipt.png"))]))
        #expect(state.items.count == 1)
        #expect(state.phase == .preparing)
        let ticket = try #require(state.active)
        #expect(commands == [.run(.prepare(ticket, state.bundle), id: .operation)])
    }

    @Test("Ending is terminal and rejects late completions and reopening the same store")
    func ending() {
        var state = ReceiptState(bundle: URL(filePath: "/model"))
        _ = ReceiptFeature.reduce(&state, .appeared)
        let commands = ReceiptFeature.reduce(&state, .ended)
        #expect(commands.first == .cancelAll)
        #expect(state.ended)
        let ended = state
        #expect(ReceiptFeature.reduce(&state, .appeared).isEmpty)
        #expect(ReceiptFeature.reduce(&state, .add([ReceiptInput(url: URL(filePath: "/late.png"))])).isEmpty)
        #expect(state == ended)
    }
}
