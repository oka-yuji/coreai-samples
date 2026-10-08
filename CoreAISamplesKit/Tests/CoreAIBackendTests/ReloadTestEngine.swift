import Foundation
import FoundationModels
import Synchronization
import Testing

@_spi(Verification) @testable import CoreAIBackend

final class ReloadTestControl: Sendable {
    struct State {
        var loads = 0
        var failNextLoad = false
        var failNextRead = false
        var failNextStep = false
        var blockNextStep = false
        var releaseStep: AsyncStream<Void>.Continuation?
        var blockNextLoad = false
        var blockNextRead = false
        var releaseRead: AsyncStream<Void>.Continuation?
        var releaseLoad: AsyncStream<Void>.Continuation?
        var liveEngines: Set<Int> = []
        var liveCheckpoints: Set<Int> = []
        var nextCheckpoint = 0
        var footprint: UInt64 = 0
        var events: [String] = []
        var generationTokens: [Int] = [1, 3]
    }

    let state = Mutex(State())
    let loading = AsyncStream<Int>.makeStream()
    let reading = AsyncStream<Int>.makeStream()

    func make(_ bundle: ModelBundle, _ contextLimit: Int) async throws -> EngineBox {
        let gate = AsyncStream<Void>.makeStream()
        let (id, block, fail, released) = state.withLock { state in
            state.loads += 1
            let id = state.loads
            let block = state.blockNextLoad
            let fail = state.failNextLoad
            state.blockNextLoad = false
            state.failNextLoad = false
            if block { state.releaseLoad = gate.continuation }
            state.events.append("load:\(id)")
            return (id, block, fail, state.liveEngines.isEmpty && state.liveCheckpoints.isEmpty)
        }
        #expect(released, "The loader retained an old engine or checkpoint")
        if block {
            loading.continuation.yield(id)
            var iterator = gate.stream.makeAsyncIterator()
            _ = await iterator.next()
            #expect(!Task.isCancelled, "The reload factory must be shielded until creation finishes")
        }
        if fail { throw ReloadTestFailure.load }
        return EngineBox(ReloadTestEngine(id: id, control: self, contextLimit: contextLimit))
    }

    func resumeLoad() {
        let continuation = state.withLock { state in
            defer { state.releaseLoad = nil }
            return state.releaseLoad
        }
        continuation?.finish()
    }

    func resumeRead() {
        let continuation = state.withLock { state in
            defer { state.releaseRead = nil }
            return state.releaseRead
        }
        continuation?.finish()
    }

    func resumeStep() {
        let continuation = state.withLock { state in
            defer { state.releaseStep = nil }
            return state.releaseStep
        }
        continuation?.finish()
    }

    func footprint() -> UInt64 { state.withLock { $0.footprint } }
    var loads: Int { state.withLock { $0.loads } }
    var events: [String] { state.withLock { $0.events } }
}

enum ReloadTestFailure: Error {
    case load
    case read
    case step
}

final class ReloadTestEngine: RuntimeEngine {
    private let id: Int
    private let control: ReloadTestControl
    private let contextLimit: Int
    private var outputIndex = 0
    private(set) var tokens: [Int32] = []
    private(set) var stats = Engine.Stats()
    private(set) var hasLogits = false
    let isHybrid = false
    let loadStats = Engine.LoadStats(cached: true, specializeSeconds: 0, totalSeconds: 0)

    init(id: Int, control: ReloadTestControl, contextLimit: Int) {
        self.id = id
        self.control = control
        self.contextLimit = contextLimit
        control.state.withLock { $0.liveEngines.insert(id) }
    }

    deinit {
        control.state.withLock {
            $0.liveEngines.remove(id)
            $0.events.append("released-engine:\(id)")
        }
    }

    func logits() -> [Float16] {
        var values = Array(repeating: Float16(-10), count: 4)
        let token = control.state.withLock { $0.generationTokens[min(outputIndex, $0.generationTokens.count - 1)] }
        values[token] = 10
        return values
    }

    func withLogits<Result>(_ body: (UnsafeBufferPointer<Float16>) -> Result) -> Result {
        logits().withUnsafeBufferPointer(body)
    }

    func checkpoint() -> Engine.Checkpoint {
        let key = control.state.withLock { state in
            state.nextCheckpoint += 1
            state.liveCheckpoints.insert(state.nextCheckpoint)
            return state.nextCheckpoint
        }
        let pointer = UnsafeMutableRawPointer.allocate(byteCount: 32_768, alignment: 8)
        pointer.initializeMemory(as: UInt8.self, repeating: 0, count: 32_768)
        let data = Data(
            bytesNoCopy: pointer, count: 32_768,
            deallocator: .custom { [control] pointer, _ in
                pointer.deallocate()
                control.state.withLock {
                    $0.liveCheckpoints.remove(key)
                    $0.events.append("released-checkpoint:\(key)")
                }
            })
        return Engine.Checkpoint(tokens: tokens, fixedStates: [data], logits: logits())
    }

    func restore(_ checkpoint: Engine.Checkpoint) throws {
        tokens = checkpoint.tokens
        outputIndex = 0
        hasLogits = true
        control.state.withLock { $0.events.append("restore:\(id)") }
    }

    func reset() {
        tokens = []
        hasLogits = false
    }

    func unload() {
        reset()
        control.state.withLock { $0.events.append("unload:\(id)") }
    }

    func prefill(_ input: [Int32], usePrefillFunction: Bool) async throws {
        let gate = AsyncStream<Void>.makeStream()
        let (fail, block) = control.state.withLock { state in
            let block = state.blockNextRead
            state.blockNextRead = false
            if block { state.releaseRead = gate.continuation }
            let fail = state.failNextRead
            state.failNextRead = false
            state.events.append("read:\(id)")
            return (fail, block)
        }
        if block {
            control.reading.continuation.yield(id)
            var iterator = gate.stream.makeAsyncIterator()
            _ = await iterator.next()
            try Task.checkCancellation()
        }
        if fail { throw ReloadTestFailure.read }
        guard tokens.count + input.count <= contextLimit else {
            throw EngineError.contextOverflow(needed: tokens.count + input.count, limit: contextLimit)
        }
        tokens += input
        stats.prefillTokens += input.count
        outputIndex = 0
        hasLogits = true
    }

    func step(_ token: Int32) async throws {
        let gate = AsyncStream<Void>.makeStream()
        let (fail, block) = control.state.withLock { state in
            let pair = (state.failNextStep, state.blockNextStep)
            state.failNextStep = false
            state.blockNextStep = false
            if pair.1 { state.releaseStep = gate.continuation }
            return pair
        }
        if block {
            var iterator = gate.stream.makeAsyncIterator()
            _ = await iterator.next()
            try Task.checkCancellation()
        }
        if fail { throw ReloadTestFailure.step }
        tokens.append(token)
        stats.decodeTokens += 1
        outputIndex += 1
    }
}

struct ReloadTestModel {
    let model: LocalLanguageModel
    let runtime: Runtime
    let folder: URL
    let control: ReloadTestControl

    static func make(policy: LocalLanguageModel.ReloadPolicy, limit: Int = 128) async throws -> Self {
        let folder = try tinyBundle()
        let config = [
            "tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|im_end|>",
            "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
        ]
        try JSONSerialization.data(withJSONObject: config).write(
            to: folder.appending(path: "tokenizer/tokenizer_config.json"))
        let control = ReloadTestControl()
        let runtime = try await Runtime(
            bundle: ModelBundle(url: folder), contextLimit: limit, overflow: .error,
            settings: .init(), reloadPolicy: policy, makeEngine: control.make, footprint: control.footprint)
        return Self(
            model: LocalLanguageModel(runtime: runtime, bundle: folder), runtime: runtime, folder: folder,
            control: control)
    }

    func request(_ prompt: String = "a") async throws -> LanguageModelSession.Response<String> {
        try await LanguageModelSession(model: model).respond(
            to: prompt, options: .init(samplingMode: .greedy, maximumResponseTokens: 8))
    }
}

func directReloadRequest(_ transcript: Transcript) -> LanguageModelExecutorGenerationRequest {
    LanguageModelExecutorGenerationRequest(
        id: UUID(), transcript: transcript, enabledTools: [],
        generationOptions: .init(samplingMode: .greedy, maximumResponseTokens: 1),
        contextOptions: .init(), metadata: [:])
}
