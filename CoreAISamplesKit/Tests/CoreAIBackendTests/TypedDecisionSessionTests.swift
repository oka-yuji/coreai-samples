import Foundation
import Synchronization
import Testing

@testable import CoreAIBackend

private enum SessionFailure: Error { case expected }

private final class SessionBackbone: TypedDecisionBackbone {
    let operation: @Sendable ([Int], Int) async throws -> [Float]
    let released: @Sendable () -> Void

    init(
        operation: @escaping @Sendable ([Int], Int) async throws -> [Float] = { ids, capacity in
            [Float(ids.count), Float(capacity)]
        }, released: @escaping @Sendable () -> Void = {}
    ) {
        self.operation = operation
        self.released = released
    }

    deinit { released() }

    func forward(_ ids: [Int], capacity: Int) async throws -> [Float] {
        try await operation(ids, capacity)
    }
}

private actor SessionFactory {
    private(set) var calls = 0
    private let failingCalls: Set<Int>

    init(failingCalls: Set<Int> = []) {
        self.failingCalls = failingCalls
    }

    func load() throws -> any TypedDecisionBackbone {
        calls += 1
        if failingCalls.contains(calls) { throw SessionFailure.expected }
        return SessionBackbone()
    }
}

@Suite(.timeLimit(.minutes(1)))
struct TypedDecisionSessionTests {
    @Test("An unchanged shape reuses its backbone for one hundred calls")
    func sameShape() async throws {
        let factory = SessionFactory()
        let session = TypedDecisionSession(backbone: SessionBackbone()) { try await factory.load() }
        for _ in 0..<100 {
            #expect(try await session.forward([1], capacity: 256) == [1, 256])
        }
        #expect(await factory.calls == 0)
    }

    @Test("The old backbone is released before reloading for the fourth shape change")
    func releaseBeforeReload() async throws {
        let releases = Mutex(0)
        let reloads = Mutex(0)
        let session = TypedDecisionSession(backbone: SessionBackbone(released: { releases.withLock { $0 += 1 } })) {
            #expect(releases.withLock { $0 } == 1)
            reloads.withLock { $0 += 1 }
            return SessionBackbone()
        }
        for count in [1, 2, 3, 4] {
            #expect(try await session.forward(Array(repeating: 1, count: count), capacity: 256) == [Float(count), 256])
        }
        #expect(reloads.withLock { $0 } == 0 && releases.withLock { $0 } == 0)
        #expect(try await session.forward([1], capacity: 256) == [1, 256])
        #expect(reloads.withLock { $0 } == 1 && releases.withLock { $0 } == 1)
    }

    @Test("Returning to a prior shape and changing only capacity both count toward recycling")
    func shapeIdentityAndRepeatedChanges() async throws {
        let factory = SessionFactory()
        let session = TypedDecisionSession(backbone: SessionBackbone()) { try await factory.load() }
        for capacity in [256, 512, 256, 512, 256, 256, 512, 256, 512, 256] {
            #expect(try await session.forward([1], capacity: capacity) == [1, Float(capacity)])
        }
        #expect(await factory.calls == 2)
    }

    @Test("A failed inference discards the backbone and the next request reloads it")
    func failedForward() async throws {
        let factory = SessionFactory()
        let releases = Mutex(0)
        let session = TypedDecisionSession(
            backbone: SessionBackbone(
                operation: { _, _ in throw SessionFailure.expected }, released: { releases.withLock { $0 += 1 } })
        ) { try await factory.load() }
        await #expect(throws: SessionFailure.self) { try await session.forward([1], capacity: 256) }
        #expect(releases.withLock { $0 } == 1)
        #expect(try await session.forward([1], capacity: 256) == [1, 256])
        #expect(await factory.calls == 1)
    }

    @Test("A failed reload leaves no usable old backbone and the next request retries")
    func failedReload() async throws {
        let factory = SessionFactory(failingCalls: [1])
        let session = TypedDecisionSession(backbone: SessionBackbone()) { try await factory.load() }
        for count in 1...4 { _ = try await session.forward(Array(repeating: 1, count: count), capacity: 256) }
        await #expect(throws: SessionFailure.self) { try await session.forward([1], capacity: 256) }
        #expect(try await session.forward([1], capacity: 256) == [1, 256])
        #expect(await factory.calls == 2)
    }

    @Test("A cancelled inference keeps the session busy until its uncooperative backbone finishes")
    func cancellationWhileRunning() async throws {
        let gate = TypedDecisionGate()
        let factory = SessionFactory()
        let (entered, continuation) = AsyncStream.makeStream(of: Void.self)
        let calls = Mutex(0)
        let session = TypedDecisionSession(
            backbone: SessionBackbone(operation: { _, _ in
                let count = calls.withLock { value in
                    value += 1
                    return value
                }
                if count == 1 {
                    continuation.yield()
                    await gate.hold()
                }
                return [9]
            })
        ) { try await factory.load() }
        let task = Task { try await session.forward([1], capacity: 256) }
        var iterator = entered.makeAsyncIterator()
        let arrival = await iterator.next()
        task.cancel()
        await #expect(throws: TypedDecisionError.busy) { try await session.forward([1], capacity: 256) }
        await gate.open()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(arrival != nil)
        #expect(try await session.forward([1], capacity: 256) == [1, 256])
        #expect(await factory.calls == 1)
    }

    @Test("A cancelled reload stays busy until it finishes and never forwards the cancelled input")
    func cancellationWhileReloading() async throws {
        let gate = TypedDecisionGate()
        let (entered, continuation) = AsyncStream.makeStream(of: Void.self)
        let calls = Mutex(0)
        let reloads = Mutex(0)
        let session = TypedDecisionSession(backbone: SessionBackbone()) {
            let count = reloads.withLock { value in
                value += 1
                return value
            }
            if count == 1 {
                continuation.yield()
                await gate.hold()
            }
            return SessionBackbone(operation: { _, _ in
                calls.withLock { $0 += 1 }
                return [7]
            })
        }
        for count in 1...4 { _ = try await session.forward(Array(repeating: 1, count: count), capacity: 256) }
        let task = Task { try await session.forward([1], capacity: 256) }
        var iterator = entered.makeAsyncIterator()
        let arrival = await iterator.next()
        task.cancel()
        await #expect(throws: TypedDecisionError.busy) { try await session.forward([1], capacity: 256) }
        await gate.open()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(arrival != nil && calls.withLock { $0 } == 0)
        #expect(try await session.forward([1], capacity: 256) == [7])
        #expect(calls.withLock { $0 } == 1)
    }

    @Test("A cache-miss fallback accepts only the same intact bundle version")
    func fallbackRevalidatesBundle() async throws {
        let root = try TypedDecisionFakeBundle.create()
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
        let document = TypedDecisionFakeBundle.document()
        try TypedDecisionFakeBundle.write(document, to: root)
        let bundle = try TypedDecisionBundle(url: root)
        let calls = Mutex(0)
        let loader = TypedDecisionLoader<Int>(
            bundle: bundle, bookmark: Data(), resolve: { _ in nil },
            fresh: { verified in
                #expect(verified.version == bundle.version)
                calls.withLock { $0 += 1 }
                return (7, Data())
            })
        #expect(try await loader.load() == 7)
        let changed = TypedDecisionFakeBundle.setting(1.3, at: ["temperatures", "per_kind", "noul"], in: document)
        try TypedDecisionFakeBundle.write(changed, to: root)
        await #expect(throws: TypedDecisionError.self) { try await loader.load() }
        try TypedDecisionFakeBundle.write(document, to: root)
        try Data("edit".utf8).write(to: root.appending(path: "fixture.aimodel/main.mlirb"))
        await #expect(throws: TypedDecisionError.self) { try await loader.load() }
        #expect(calls.withLock { $0 } == 1)
    }

    @Test("A fresh load updates the bookmark and a later cache hit does not reread the source bundle")
    func bookmarkIsRefreshed() async throws {
        let root = try TypedDecisionFakeBundle.create()
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
        try TypedDecisionFakeBundle.write(TypedDecisionFakeBundle.document(), to: root)
        let bundle = try TypedDecisionBundle(url: root)
        let resolved = Mutex<[Data]>([])
        let calls = Mutex(0)
        let loader = TypedDecisionLoader<Int>(
            bundle: bundle, bookmark: Data([1]),
            resolve: { bookmark in
                resolved.withLock { $0.append(bookmark) }
                return bookmark == Data([2]) ? (7, bookmark) : nil
            },
            fresh: { _ in
                calls.withLock { $0 += 1 }
                return (7, Data([2]))
            })
        #expect(try await loader.load() == 7)
        try FileManager.default.removeItem(at: root.appending(path: "decision.json"))
        #expect(try await loader.load() == 7)
        #expect(calls.withLock { $0 } == 1)
        #expect(resolved.withLock { $0 } == [Data([1]), Data([2])])
    }

    @Test("A malformed cache bookmark propagates its error without loading a fresh model")
    func cacheErrorDoesNotFallBack() async throws {
        let root = try TypedDecisionFakeBundle.create()
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
        try TypedDecisionFakeBundle.write(TypedDecisionFakeBundle.document(), to: root)
        let bundle = try TypedDecisionBundle(url: root)
        let calls = Mutex(0)
        let loader = TypedDecisionLoader<Int>(
            bundle: bundle, bookmark: Data(), resolve: { _ in throw SessionFailure.expected },
            fresh: { _ in
                calls.withLock { $0 += 1 }
                return (7, Data())
            })
        try FileManager.default.removeItem(at: root.appending(path: "decision.json"))
        await #expect(throws: SessionFailure.self) { try await loader.load() }
        #expect(calls.withLock { $0 } == 0)
    }
}
