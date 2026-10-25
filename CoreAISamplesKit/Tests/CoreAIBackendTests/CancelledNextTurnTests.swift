import Foundation
import FoundationModels
import Synchronization
import Testing

@_spi(Verification) @testable import CoreAIBackend

final class EngineCallLog: Sendable {
    enum Call: Sendable {
        case reset
        case prefill([Int32], usesPrefillFunction: Bool)
        case step(Int32)
        case checkpoint
        case restore(Int)
        case restorePrefix(Int)
        case unload
        case sample([Float16])
    }

    private struct State {
        var calls: [Call] = []
        var checkpoints: [Engine.Checkpoint] = []
    }

    private let state = Mutex(State())

    var calls: [Call] { state.withLock { $0.calls } }

    func record(_ call: Call) {
        state.withLock { $0.calls.append(call) }
    }

    func recordCheckpoint(_ point: Engine.Checkpoint) {
        state.withLock { state in
            state.calls.append(.checkpoint)
            state.checkpoints.append(point)
        }
    }

    func recordRestore(_ point: Engine.Checkpoint) {
        state.withLock { state in
            let index = state.checkpoints.firstIndex {
                $0.tokens == point.tokens && $0.fixedStates == point.fixedStates
            }
            state.calls.append(index.map(Call.restore) ?? .restorePrefix(point.tokens.count))
        }
    }

    func replay(on engine: Engine) async throws -> [[Float16]] {
        var points: [Engine.Checkpoint] = []
        var samples: [[Float16]] = []
        for call in calls {
            switch call {
            case .reset: engine.reset()
            case .prefill(let tokens, let usesPrefillFunction):
                try await engine.prefill(tokens, usePrefillFunction: usesPrefillFunction)
            case .step(let token): try await engine.step(token)
            case .checkpoint: points.append(engine.checkpoint())
            case .restore(let index): try engine.restore(points[index])
            case .restorePrefix(let count):
                try engine.restore(Engine.Checkpoint(tokens: Array(engine.tokens.prefix(count)), fixedStates: []))
            case .unload: engine.unload()
            case .sample: samples.append(engine.logits())
            }
        }
        return samples
    }
}

final class RecordingEngine: RuntimeEngine {
    private let engine: Engine
    private let log: EngineCallLog

    init(_ engine: Engine, log: EngineCallLog) {
        self.engine = engine
        self.log = log
    }

    var tokens: [Int32] { engine.tokens }
    var stats: Engine.Stats { engine.stats }
    var loadStats: Engine.LoadStats { engine.loadStats }
    var isHybrid: Bool { engine.isHybrid }
    var hasLogits: Bool { engine.hasLogits }
    var measuresPreparation: Bool {
        get { engine.measuresPreparation }
        set { engine.measuresPreparation = newValue }
    }

    func logits() -> [Float16] { engine.logits() }

    func withLogits<Result>(_ body: (UnsafeBufferPointer<Float16>) -> Result) -> Result {
        log.record(.sample(engine.logits()))
        return engine.withLogits(body)
    }

    func checkpoint() -> Engine.Checkpoint {
        let point = engine.checkpoint()
        log.recordCheckpoint(point)
        return point
    }

    func restore(_ checkpoint: Engine.Checkpoint) throws {
        try engine.restore(checkpoint)
        log.recordRestore(checkpoint)
    }

    func reset() {
        engine.reset()
        log.record(.reset)
    }

    func unload() {
        engine.unload()
        log.record(.unload)
    }

    func prefill(_ input: [Int32], usePrefillFunction: Bool) async throws {
        try await engine.prefill(input, usePrefillFunction: usePrefillFunction)
        log.record(.prefill(input, usesPrefillFunction: usePrefillFunction))
    }

    func prefill(_ input: [Int32], usePrefillFunction: Bool, onProgress: @Sendable (Int) -> Void) async throws {
        try await engine.prefill(input, usePrefillFunction: usePrefillFunction, onProgress: onProgress)
        log.record(.prefill(input, usesPrefillFunction: usePrefillFunction))
    }

    func step(_ token: Int32) async throws {
        try await engine.step(token)
        log.record(.step(token))
    }
}

extension ModelIntegrationTests {
    @Test(
        "After a cancelled request on a long preamble, the next request in the same session matches a fresh replay of the same engine calls",
        .enabled(if: !bundlePaths.isEmpty), arguments: bundlePaths)
    func nextRequestAfterCancellationMatchesAFreshReplay(path: String) async throws {
        let url = URL(filePath: path)
        let bundle = try ModelBundle(url: url)
        let log = EngineCallLog()
        let runtime = try await Runtime(
            bundle: bundle, contextLimit: ModelTestSettings.contextLimit, overflow: .error, settings: .init(),
            reloadPolicy: .init(readInterval: 0, footprintLimitBytes: 0),
            makeEngine: { bundle, limit in
                EngineBox(RecordingEngine(try await Engine(bundle: bundle, contextLimit: limit), log: log))
            })
        let model = LocalLanguageModel(runtime: runtime, bundle: url)
        let resource = Bundle.module.url(forResource: "long-preamble", withExtension: "txt", subdirectory: "Resources")
        let session = LanguageModelSession(
            model: model, instructions: try String(contentsOf: #require(resource), encoding: .utf8))
        try await session.respond(
            to: "Which floor is Room 21 on? Answer in one sentence.",
            options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 64),
            contextOptions: ModelTestSettings.context)
        _ = try await cancelAfterFirstFragment(
            session, prompt: "Write a long story about a dragon.", maximumTokens: 2048,
            waitUntilFinished: { _ = await model.processedTokens() })
        let before = log.calls.count
        let after = try await session.respond(
            to: "Describe every room on floor 2 in detail, one paragraph per room.",
            options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 32),
            contextOptions: ModelTestSettings.context)
        let processed = await model.processedTokens()
        let calls = log.calls
        let original = calls.compactMap { call -> [Float16]? in
            if case .sample(let logits) = call { logits } else { nil }
        }
        let nextSamples = calls[before...].filter { if case .sample = $0 { true } else { false } }.count
        let restores = calls[before...].filter { if case .restore = $0 { true } else { false } }.count
        let replay = try await Engine(bundle: bundle, contextLimit: ModelTestSettings.contextLimit)
        let replayed = try await log.replay(on: replay)
        let differing = zip(original, replayed).enumerated().filter {
            $0.element.0.map(\.bitPattern) != $0.element.1.map(\.bitPattern)
        }.map(\.offset)
        #expect(after.usage.output.totalTokenCount == 32)
        #expect(nextSamples == 32)
        #expect(replay.tokens == processed)
        #expect(replayed.count == original.count)
        #expect(differing.isEmpty, "samples that differ from the replay: \(differing)")
        print(
            "cancel-next-replay chunk=\(replay.prefillChunkSize) processed=\(processed.count)"
                + " cached=\(after.usage.input.cachedTokenCount) prompt=\(after.usage.input.totalTokenCount)"
                + " restores=\(restores) samples=\(original.count) next=\(nextSamples) differing=\(differing.count)")
        try await ModelTestSettings.checkState(model, label: "cancel-next-replay")
    }
}
