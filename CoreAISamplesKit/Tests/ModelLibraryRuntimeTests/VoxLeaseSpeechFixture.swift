import FeatureStore
import Foundation
import SpeechCore
import SpeechFeature

@testable import SpeechRuntime

final class VoxLeaseTestResource: Sendable {
    let signals: VoxLeaseSignals
    init(signals: VoxLeaseSignals) { self.signals = signals }
    deinit { signals.record("resource-destroyed") }
}

actor VoxLeasePhysicalSpeech: SpeechGenerating {
    nonisolated let signals: VoxLeaseSignals
    nonisolated let opening: VoxLeaseGate
    nonisolated let generating: VoxLeaseGate
    nonisolated let releasing: VoxLeaseGate
    private var resource: VoxLeaseTestResource?

    init(open: Bool = false, generate: Bool = false, release: Bool = false) {
        signals = VoxLeaseSignals()
        opening = VoxLeaseGate(closed: open)
        generating = VoxLeaseGate(closed: generate)
        releasing = VoxLeaseGate(closed: release)
    }
    nonisolated func unblock() {
        opening.open()
        generating.open()
        releasing.open()
    }

    func open(_ url: URL) async throws -> SpeechModel {
        resource = VoxLeaseTestResource(signals: signals)
        signals.record("open-enter")
        await opening.waitIgnoringCancellation()
        signals.record("open-leave")
        return SpeechModel(name: "Synthetic voice", directory: url, supportsReferenceAudio: true)
    }
    func generate(text: String, seed: UInt64) async throws -> SpeechAudio {
        try await generate(text: text, seed: seed, reference: nil)
    }
    func generate(text: String, seed: UInt64, reference: SpeechReferenceAudio?) async throws -> SpeechAudio {
        if resource == nil { resource = VoxLeaseTestResource(signals: signals) }
        signals.record("generate-enter")
        if let reference {
            signals.record("reference-\(reference.samples.count)-\(text)-\(seed)")
        }
        await generating.waitIgnoringCancellation()
        signals.record("generate-leave")
        return SpeechAudio(wav: Data([0]), audioSeconds: 1, elapsedSeconds: 1, normalizedText: text)
    }
    func release() async {
        signals.record("release-enter")
        await releasing.waitIgnoringCancellation()
        resource = nil
        signals.record("release-leave")
    }
}

@MainActor
final class VoxLeaseSpeechService: SpeechService {
    let engine: any SpeechGenerating
    init(_ engine: any SpeechGenerating) { self.engine = engine }
    func open(path: String) async throws -> SpeechModel { try await engine.open(URL(filePath: path)) }
    func generate(text: String, seed: UInt64) async throws -> SpeechAudio {
        try await engine.generate(text: text, seed: seed)
    }
    func generate(text: String, seed: UInt64, reference: SpeechReferenceAudio?) async throws -> SpeechAudio {
        try await engine.generate(text: text, seed: seed, reference: reference)
    }
    func play(_ data: Data) async throws { throw VoxLeaseProbeError(description: "unused playback") }
    func stop() async { await engine.release() }

    func makeStore(_ signals: VoxLeaseSignals) -> Store<SpeechFeature> {
        let runner = SpeechRunner(service: self)
        return Store(initialState: SpeechState()) { effect, send in
            await runner.run(effect, send: send)
            if case .open = effect { signals.record("speech-open-ack") }
            if case .stop = effect { signals.record("speech-stop-ack") }
        }
    }
}
