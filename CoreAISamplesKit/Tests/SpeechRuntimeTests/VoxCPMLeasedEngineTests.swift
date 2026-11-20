import FeatureStore
import Foundation
import Observation
import SpeechCore
import SpeechFeature
import Synchronization
import Testing

@testable import SpeechRuntime

private struct LeaseRefusal: Error {}

private final class LeaseTrace: Sendable {
    private struct Waiter {
        let id: UUID
        let label: String
        let count: Int
        let continuation: AsyncStream<Void>.Continuation
    }
    private struct Book {
        var log: [String] = []
        var waiters: [Waiter] = []
    }
    private let book = Mutex(Book())
    var log: [String] { book.withLock { $0.log } }

    func record(_ label: String) {
        let ready = book.withLock { book in
            book.log.append(label)
            let count = book.log.count { $0 == label }
            let ready = book.waiters.filter { $0.label == label && $0.count <= count }
            book.waiters.removeAll { $0.label == label && $0.count <= count }
            return ready.map(\.continuation)
        }
        for waiter in ready {
            waiter.yield(())
            waiter.finish()
        }
    }

    func wait(_ label: String, _ count: Int = 1) async throws {
        try Task.checkCancellation()
        let id = UUID()
        let (stream, continuation) = AsyncStream<Void>.makeStream()
        let arrived = book.withLock { book in
            if book.log.count(where: { $0 == label }) >= count { return true }
            book.waiters.append(Waiter(id: id, label: label, count: count, continuation: continuation))
            return false
        }
        defer {
            book.withLock { $0.waiters.removeAll { $0.id == id } }
            continuation.finish()
        }
        if !arrived {
            var iterator = stream.makeAsyncIterator()
            guard await iterator.next() != nil else { throw CancellationError() }
        }
        try Task.checkCancellation()
    }

    func count(_ label: String) -> Int { log.count { $0 == label } }

    func precedes(_ labels: String...) -> Bool {
        let log = log
        let positions = labels.map { log.firstIndex(of: $0) }
        guard positions.allSatisfy({ $0 != nil }) else { return false }
        let found = positions.compactMap { $0 }
        return zip(found, found.dropFirst()).allSatisfy { $0 < $1 }
    }
}

private final class LeaseLatch: Sendable {
    private struct State {
        var open: Bool
        var waiters: [CheckedContinuation<Void, Never>] = []
    }
    private let state: Mutex<State>
    init(open: Bool) { state = Mutex(State(open: open)) }

    func wait() async {
        await withCheckedContinuation { continuation in
            let open = state.withLock { state in
                guard !state.open else { return true }
                state.waiters.append(continuation)
                return false
            }
            if open { continuation.resume() }
        }
    }

    func open() {
        let waiting = state.withLock { state in
            state.open = true
            defer { state.waiters.removeAll() }
            return state.waiters
        }
        for waiter in waiting { waiter.resume() }
    }
}

private actor TraceEngine: SpeechGenerating {
    nonisolated let trace: LeaseTrace
    nonisolated let generating: LeaseLatch
    nonisolated let releasing: LeaseLatch
    private let openFailure: SpeechError?

    init(trace: LeaseTrace, openFailure: SpeechError? = nil, generates: Bool = true, releases: Bool = true) {
        self.trace = trace
        self.openFailure = openFailure
        generating = LeaseLatch(open: generates)
        releasing = LeaseLatch(open: releases)
    }

    func open(_ directory: URL) async throws -> SpeechModel {
        trace.record("open:" + directory.lastPathComponent)
        if let openFailure { throw openFailure }
        return SpeechModel(name: "Trace voice", directory: directory, supportsReferenceAudio: true)
    }

    func generate(text: String, seed: UInt64) async throws -> SpeechAudio {
        try await generate(text: text, seed: seed, reference: nil)
    }

    func generate(text: String, seed: UInt64, reference: SpeechReferenceAudio?) async throws -> SpeechAudio {
        trace.record("generate:\(text):\(seed):\(reference?.samples.count ?? 0)")
        await generating.wait()
        if text == "too long" { throw SpeechError.textExceedsLimit(count: 200, limit: 144) }
        return SpeechAudio(wav: Data([1]), audioSeconds: 1, elapsedSeconds: 1, normalizedText: text)
    }

    func release() async {
        trace.record("release-enter")
        await releasing.wait()
        trace.record("release-leave")
    }
}

private actor TraceBank {
    let trace: LeaseTrace
    private(set) var requests: [Int?] = []
    private var generations: [String: Int] = [:]
    private var held: [Int: String] = [:]
    private var next = 0
    private var delay: LeaseLatch?

    init(trace: LeaseTrace) { self.trace = trace }

    func delayNext(_ latch: LeaseLatch) { delay = latch }
    func replace(_ directory: URL) { generations[directory.lastPathComponent, default: 1] += 1 }
    func held(_ directory: URL) -> Int { held.values.count { $0 == directory.lastPathComponent } }

    func acquire(_ directory: URL, _ expected: Int?) async throws -> (
        registrationGeneration: Int, release: @Sendable () async -> Void
    ) {
        let name = directory.lastPathComponent
        let generation = generations[name, default: 1]
        requests.append(expected)
        guard expected == nil || expected == generation else { throw LeaseRefusal() }
        let id = next
        next += 1
        held[id] = name
        trace.record("acquired:" + name)
        if let latch = delay {
            delay = nil
            await latch.wait()
        }
        return (generation, { await self.drop(id) })
    }

    private func drop(_ id: Int) {
        guard let name = held.removeValue(forKey: id) else { return trace.record("double-return") }
        trace.record("returned:" + name)
    }
}

@Suite(.timeLimit(.minutes(1)))
struct VoxCPMLeasedEngineTests {
    private let first = URL(filePath: "/synthetic/voxcpm2-reference_fp32", directoryHint: .isDirectory)
    private let second = URL(filePath: "/synthetic/voxcpm2-fp16", directoryHint: .isDirectory)

    private func leased(_ inner: TraceEngine, _ bank: TraceBank) -> VoxCPMLeasedEngine {
        let trace = inner.trace
        return VoxCPMLeasedEngine(
            engine: inner, acquire: { try await bank.acquire($0, $1) },
            didRequestRelease: { trace.record("release-requested") })
    }

    @Test("Concurrent releases wait for an uncooperative generation and the inner release before one lease return")
    func joinedRelease() async throws {
        let trace = LeaseTrace()
        let inner = TraceEngine(trace: trace, generates: false, releases: false)
        let bank = TraceBank(trace: trace)
        let engine = leased(inner, bank)
        let unblock: @Sendable () -> Void = {
            inner.generating.open()
            inner.releasing.open()
        }
        _ = try await engine.open(first)
        #expect(trace.precedes("acquired:voxcpm2-reference_fp32", "open:voxcpm2-reference_fp32"))
        try await withTaskCancellationHandler {
            async let generation = engine.generate(text: "hello", seed: 7)
            defer { unblock() }
            try await trace.wait("generate:hello:7:0")
            async let leaving: Void = engine.release()
            defer { unblock() }
            try await trace.wait("release-requested")
            async let joining: Void = engine.release()
            defer { unblock() }
            try await trace.wait("release-requested", 2)
            await #expect(throws: SpeechError.busy) { try await engine.generate(text: "late", seed: 1) }
            inner.generating.open()
            try await trace.wait("release-enter")
            let protected = await bank.held(first)
            #expect(protected == 1 && trace.count("returned:voxcpm2-reference_fp32") == 0)
            inner.releasing.open()
            _ = try await generation
            await leaving
            await joining
        } onCancel: {
            unblock()
        }
        let remaining = await bank.held(first)
        #expect(remaining == 0 && trace.count("release-enter") == 1)
        #expect(trace.count("returned:voxcpm2-reference_fp32") == 1 && trace.count("double-return") == 0)
        #expect(trace.precedes("release-leave", "returned:voxcpm2-reference_fp32"))
        #expect(trace.count("generate:late:1:0") == 0)
    }

    @Test("An acquisition that arrives after release begins is returned without opening the model")
    func lateAcquisition() async throws {
        let trace = LeaseTrace()
        let inner = TraceEngine(trace: trace)
        let bank = TraceBank(trace: trace)
        let delivery = LeaseLatch(open: false)
        await bank.delayNext(delivery)
        let engine = leased(inner, bank)
        let opened = try await withTaskCancellationHandler {
            async let opening = engine.open(first)
            defer { delivery.open() }
            try await trace.wait("acquired:voxcpm2-reference_fp32")
            async let closing: Void = engine.release()
            defer { delivery.open() }
            try await trace.wait("release-requested")
            delivery.open()
            var opened = false
            do {
                _ = try await opening
                opened = true
            } catch {}
            await closing
            return opened
        } onCancel: {
            delivery.open()
        }
        let remaining = await bank.held(first)
        #expect(!opened && remaining == 0 && trace.count("open:voxcpm2-reference_fp32") == 0)
        #expect(trace.count("returned:voxcpm2-reference_fp32") == 1 && trace.count("double-return") == 0)
        _ = try await engine.open(first)
        #expect(trace.count("open:voxcpm2-reference_fp32") == 1)
        await engine.release()
        let released = await bank.held(first)
        #expect(released == 0 && trace.count("double-return") == 0)
    }

    @Test("A generation after stop reacquires the opened registration and a replaced registration needs a new open")
    func reacquisition() async throws {
        let trace = LeaseTrace()
        let bank = TraceBank(trace: trace)
        let engine = leased(TraceEngine(trace: trace), bank)
        _ = try await engine.open(first)
        await engine.release()
        let stopped = await bank.held(first)
        #expect(stopped == 0)
        let reference = try SpeechReferenceAudio(
            samples: Array(repeating: 0.25, count: 80_000), source: .file("voice.wav"), format: .voxcpm2)
        let generating: any SpeechGenerating = engine
        _ = try await generating.generate(text: "again", seed: 1708, reference: reference)
        let warm = await bank.held(first)
        #expect(warm == 1 && trace.count("generate:again:1708:80000") == 1)
        await engine.release()
        await bank.replace(first)
        await #expect(throws: LeaseRefusal.self) { try await engine.generate(text: "stale", seed: 1) }
        let refused = await bank.held(first)
        #expect(refused == 0 && trace.count("generate:stale:1:0") == 0)
        _ = try await engine.open(first)
        _ = try await engine.generate(text: "fresh", seed: 1)
        let requests = await bank.requests
        #expect(requests == [nil, 1, 1, nil])
        #expect(trace.count("open:voxcpm2-reference_fp32") == 2 && trace.count("generate:fresh:1:0") == 1)
        await engine.release()
        let released = await bank.held(first)
        #expect(released == 0 && trace.count("double-return") == 0)
    }

    @Test("A rejected input keeps the warm model and its execution lease until the explicit release")
    func warmRejection() async throws {
        let trace = LeaseTrace()
        let bank = TraceBank(trace: trace)
        let engine = leased(TraceEngine(trace: trace), bank)
        _ = try await engine.open(first)
        await #expect(throws: SpeechError.textExceedsLimit(count: 200, limit: 144)) {
            try await engine.generate(text: "too long", seed: 1)
        }
        _ = try await engine.generate(text: "short", seed: 1)
        let requests = await bank.requests
        let warm = await bank.held(first)
        #expect(requests == [nil] && warm == 1 && trace.count("release-enter") == 0)
        await engine.release()
        let released = await bank.held(first)
        #expect(released == 0 && trace.count("returned:voxcpm2-reference_fp32") == 1)
    }

    @Test("A failed open returns its lease after the inner release and leaves nothing to generate")
    func failedOpen() async throws {
        let trace = LeaseTrace()
        let bank = TraceBank(trace: trace)
        let engine = leased(TraceEngine(trace: trace, openFailure: .invalidAsset("Broken bundle.")), bank)
        await #expect(throws: SpeechError.invalidAsset("Broken bundle.")) { try await engine.open(first) }
        let remaining = await bank.held(first)
        #expect(remaining == 0 && trace.precedes("release-leave", "returned:voxcpm2-reference_fp32"))
        await #expect(throws: SpeechError.modelNotOpened("VoxCPM2")) {
            try await engine.generate(text: "after", seed: 1)
        }
        let requests = await bank.requests
        #expect(requests == [nil] && trace.count("double-return") == 0)
    }

    @Test("Opening another variant releases the model and returns the old lease before acquiring the new one")
    func switching() async throws {
        let trace = LeaseTrace()
        let bank = TraceBank(trace: trace)
        let engine = leased(TraceEngine(trace: trace), bank)
        _ = try await engine.open(first)
        _ = try await engine.open(second)
        #expect(
            trace.precedes(
                "open:voxcpm2-reference_fp32", "release-leave", "returned:voxcpm2-reference_fp32",
                "acquired:voxcpm2-fp16", "open:voxcpm2-fp16"))
        let old = await bank.held(first)
        let current = await bank.held(second)
        #expect(old == 0 && current == 1)
        await engine.release()
        let released = await bank.held(second)
        #expect(released == 0 && trace.count("double-return") == 0)
    }

    @MainActor
    @Test("The live VoxCPM2 store asks for the execution lease before it opens the selected folder")
    func liveStore() async throws {
        let name = "VoxCPMLeasedEngineTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let trace = LeaseTrace()
        let store = LiveSpeechService.makeStore(
            engine: .voxcpm2, examples: [], defaults: defaults,
            acquireVoxCPM: { directory, expected in
                trace.record("acquire:\(directory.path):\(expected.map { String($0) } ?? "none")")
                throw SpeechError.invalidAsset("Execution lease refused.")
            })
        store.send(.appeared)
        store.send(.openBundle("/synthetic/voxcpm2-reference_fp32"))
        for await message in Observations({ store.state.message }) where message != nil { break }
        #expect(store.state.message == "Execution lease refused." && store.state.model == nil)
        #expect(store.state.phase == .idle && trace.log == ["acquire:/synthetic/voxcpm2-reference_fp32:none"])
        #expect(defaults.string(forKey: SpeechEngine.voxcpm2.modelPathKey) == nil)
    }
}
