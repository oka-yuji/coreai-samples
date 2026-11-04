import Foundation
import Testing

@testable import CoreAIBackend

private actor TypedDecisionSessionReloads {
    private(set) var count = 0
    func record() { count += 1 }
}

extension TypedDecisionInferenceTests {
    private static func model(
        entered: AsyncStream<Void>.Continuation, reloads: TypedDecisionSessionReloads
    ) async throws -> TypedDecisionModel {
        let bundle = try TypedDecisionBundle(url: URL(filePath: typedDecisionBundle))
        let tokenizer = try await DecisionModel.loadTokenizer(bundle.tokenizerURL)
        let head = try TypedDecisionHead(url: bundle.headURL, width: bundle.hiddenSize)
        let (runtime, _) = try await TypedDecisionRuntime.load(bundle: bundle)
        let loader = TypedDecisionRuntime.loader(bundle: bundle, bookmark: runtime.bookmarkData)
        let session = TypedDecisionSession(backbone: TypedDecisionDetachedBackbone(runtime: runtime, entered: entered))
        {
            let restored = try await loader.load()
            await reloads.record()
            return TypedDecisionDetachedBackbone(runtime: restored, entered: entered)
        }
        return TypedDecisionModel(
            bundle: bundle, encode: { tokenizer.encode(text: $0, addSpecialTokens: false) }, head: head,
            backbone: session)
    }

    @Test("Cancelling a real session waits for its uncooperative run and reloads before the next identical answer")
    func cancelledRunReloadsBeforeReuse() async throws {
        let (entered, continuation) = AsyncStream.makeStream(of: Void.self)
        let reloads = TypedDecisionSessionReloads()
        let model = try await Self.model(entered: continuation, reloads: reloads)
        let row = try #require(try TypedDecisionInferenceRow.rows().first)
        var iterator = entered.makeAsyncIterator()
        let before = try await model.decide(row.request)
        #expect(await iterator.next() != nil)
        let request = TypedDecisionInferenceTests.longRequest(row)
        let count = try await model.tokenCount(for: request)
        let task = Task { try await model.decide(request) }
        #expect(await iterator.next() != nil)
        let requested = ContinuousClock.now
        task.cancel()
        await #expect(throws: TypedDecisionError.busy) { try await model.decide(row.request) }
        await #expect(throws: CancellationError.self) { try await task.value }
        let cancelledSeconds = (ContinuousClock.now - requested) / .seconds(1)
        let after = try await model.decide(row.request)
        #expect(await iterator.next() != nil)
        #expect(after.logits.map(\.bitPattern) == before.logits.map(\.bitPattern))
        #expect(after.probabilities.map(\.bitPattern) == before.probabilities.map(\.bitPattern))
        #expect(after.probabilitiesT1.map(\.bitPattern) == before.probabilitiesT1.map(\.bitPattern))
        #expect(await reloads.count == 1)
        try DecisionInferenceTests.writeMetrics(
            ["tokens": count, "cancelledSeconds": cancelledSeconds, "reloads": await reloads.count],
            name: "typed-session-cancel")
    }
}
