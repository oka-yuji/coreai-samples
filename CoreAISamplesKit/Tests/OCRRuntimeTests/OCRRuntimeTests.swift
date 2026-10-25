import Foundation
import OCRCore
import Synchronization
import Testing

@testable import OCRRuntime

private struct Call: Equatable {
    let cache: Int
    let position: Int
}

private final class FakeLog: Sendable {
    struct State {
        var visionCalls = 0
        var caches = 0
        var prefills: [Int] = []
        var steps: [Call] = []
    }

    let state = Mutex(State())
}

private enum CancellationPoint: CaseIterable, Sendable {
    case encoding
    case cache
}

private struct FakeModel: OCRModel {
    let log: FakeLog
    let blockedStep: Int
    let entered: AsyncStream<Void>.Continuation
    let release: AsyncStream<Void>
    var cancellationPoint: CancellationPoint? = nil

    var windowLimit: Int { 2_048 }
    var tokenLimit: Int { 32_768 }

    func encode(_ text: String) -> [Int32] {
        if cancellationPoint == .encoding { withUnsafeCurrentTask { $0?.cancel() } }
        return [1]
    }

    func decode(_ tokens: [Int32], skipSpecialTokens: Bool) -> String {
        tokens.map(String.init).joined(separator: " ")
    }

    func makeCache(capacity: Int) -> Int {
        if cancellationPoint == .cache { withUnsafeCurrentTask { $0?.cancel() } }
        return log.state.withLock { state in
            state.caches += 1
            return state.caches
        }
    }

    @concurrent func vision(_ input: OCRVisionInput) async throws -> [Float] {
        log.state.withLock { $0.visionCalls += 1 }
        return [Float](repeating: 0, count: input.windowCount * 16 * 4)
    }

    @concurrent func embed(_ ids: [Int32]) async throws -> [Float] {
        [Float](repeating: 0, count: ids.count * 4)
    }

    @concurrent func prefill(_ embeddings: [Float], rotary: OCRRotary, cache: inout Int) async throws -> [Float] {
        let id = cache
        log.state.withLock { $0.prefills.append(id) }
        return Self.logits(choosing: 100)
    }

    @concurrent func step(_ token: Int32, position: Int, rotary: OCRRotary, cache: inout Int) async throws
        -> [Float]
    {
        let call = Call(cache: cache, position: position)
        let index = log.state.withLock { state in
            state.steps.append(call)
            return state.steps.count - 1
        }
        if index == blockedStep {
            entered.yield(())
            for await _ in release { break }
        }
        return Self.logits(choosing: Int32(101 + index))
    }

    static func logits(choosing token: Int32) -> [Float] {
        var logits = [Float](repeating: 0, count: 151_936)
        logits[Int(token)] = 1
        return logits
    }
}

@Suite("OCR runtime ownership, cancellation and image files", .timeLimit(.minutes(1)))
struct OCRRuntimeTests {
    @Test("Cancellation during preparation prevents the next asset call", arguments: CancellationPoint.allCases)
    private func cancelledPreparationDoesNotRunAnAsset(at point: CancellationPoint) async throws {
        let log = FakeLog()
        let runtime = OCRRuntime(
            model: FakeModel(
                log: log, blockedStep: -1, entered: AsyncStream<Void>.makeStream().continuation,
                release: AsyncStream<Void>.makeStream().stream, cancellationPoint: point),
            loadStats: OCRLoadStats(assets: [], tokenizerSeconds: 0, totalSeconds: 0))
        let image = try OCRImage(width: 56, height: 56, rgb: [UInt8](repeating: 128, count: 56 * 56 * 3))
        let request = Task { try await runtime.recognize(image, instruction: "a", maxNewTokens: 3) }
        await #expect(throws: CancellationError.self) { try await request.value }
        let (visionCalls, caches, prefills) = log.state.withLock { ($0.visionCalls, $0.caches, $0.prefills) }
        #expect(visionCalls == (point == .encoding ? 0 : 1))
        #expect(caches == (point == .encoding ? 0 : 1))
        #expect(prefills.isEmpty)
    }

    @Test("Stop a cancelled request before its next step, drop its cache, and serve the next request with a new cache")
    func cancelledRequestDropsItsCache() async throws {
        let entered = AsyncStream<Void>.makeStream()
        let release = AsyncStream<Void>.makeStream()
        let log = FakeLog()
        let runtime = OCRRuntime(
            model: FakeModel(log: log, blockedStep: 1, entered: entered.continuation, release: release.stream),
            loadStats: OCRLoadStats(assets: [], tokenizerSeconds: 0, totalSeconds: 0))
        let image = try OCRImage(width: 56, height: 56, rgb: [UInt8](repeating: 128, count: 56 * 56 * 3))
        let first = Task { try await runtime.recognize(image, instruction: "a", maxNewTokens: 8) }
        var signals = entered.stream.makeAsyncIterator()
        _ = await signals.next()
        await #expect(throws: OCRRuntimeError.busy) {
            try await runtime.recognize(image, instruction: "a", maxNewTokens: 8)
        }
        first.cancel()
        release.continuation.yield(())
        await #expect(throws: CancellationError.self) { try await first.value }

        let second = try await runtime.recognize(image, instruction: "a", maxNewTokens: 3)
        let (caches, prefills, steps) = log.state.withLock { ($0.caches, $0.prefills, $0.steps) }
        let start = second.promptTokens
        #expect(caches == 2)
        #expect(prefills == [1, 2])
        #expect(
            steps == [
                Call(cache: 1, position: start), Call(cache: 1, position: start + 1),
                Call(cache: 2, position: start), Call(cache: 2, position: start + 1),
            ])
        #expect(second.tokens == [100, 103, 104] && second.finish == .maxNewTokens)
    }

    @Test("Reject a token budget beyond the bundle limit before running any asset", arguments: [0, 32_752, Int.max])
    func rejectsTokenBudget(maxNewTokens: Int) async throws {
        let log = FakeLog()
        let runtime = OCRRuntime(
            model: FakeModel(
                log: log, blockedStep: -1, entered: AsyncStream<Void>.makeStream().continuation,
                release: AsyncStream<Void>.makeStream().stream),
            loadStats: OCRLoadStats(assets: [], tokenizerSeconds: 0, totalSeconds: 0))
        let image = try OCRImage(width: 56, height: 56, rgb: [UInt8](repeating: 128, count: 56 * 56 * 3))
        await #expect(throws: OCRRuntimeError.self) {
            try await runtime.recognize(image, instruction: "a", maxNewTokens: maxNewTokens)
        }
        #expect(log.state.withLock { $0.caches == 0 && $0.prefills.isEmpty })
    }

    @Test(
        "Read PNG pixels as 8-bit RGB like Pillow, keeping the color of transparent pixels",
        arguments: [
            (
                "iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAYAAACddGYaAAAAIklEQVR4nGPgEpFjOJFi1MDIxPyf4T9DgwM7BydDVHTMfwBQ7gcHh96clQAAAABJRU5ErkJggg==",
                [10, 20, 30, 200, 100, 50, 1, 2, 3, 255, 0, 128, 7, 8, 9, 90, 91, 92] as [UInt8]
            ),
            (
                "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAAAAABX3VL4AAAADklEQVR4nGNg8GVo+A8AAzkBzbFhd+4AAAAASUVORK5CYII=",
                [0, 0, 0, 77, 77, 77, 128, 128, 128, 255, 255, 255]
            ),
        ])
    func readsPillowRGB(png: String, rgb: [UInt8]) throws {
        let data = try #require(Data(base64Encoded: png))
        #expect(try OCRImage(data: data).rgb == rgb)
    }
}
