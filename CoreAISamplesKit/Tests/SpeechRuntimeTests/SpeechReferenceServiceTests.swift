import Foundation
import SpeechCore
import Testing

@testable import SpeechRuntime

@MainActor
struct SpeechReferenceServiceTests {
    @Test("REF-CORE-02 Live and managed services forward reference samples and enforce the session gate")
    func routing() async throws {
        let engine = ReferenceServiceEngine()
        let live = LiveSpeechService(engine: engine, playback: AudioPlayback(), defaults: nil)
        let permission = ReferenceTestPermission()
        let managed = ManagedSpeechService(service: live, directory: URL(filePath: "/reference-model")) {
            permission.allowed
        }
        let reference = try SpeechReferenceAudio(
            samples: [Float](repeating: 0.25, count: 24_000), source: .recording)
        _ = try await managed.generate(text: "Example", seed: 1708, reference: reference)
        #expect(await engine.opened?.path == "/reference-model")
        #expect(await engine.reference == reference)
        #expect(await engine.requests == 1)
        permission.allowed = false
        await #expect(throws: CancellationError.self) {
            try await managed.generate(text: "Example", seed: 1708, reference: reference)
        }
        #expect(await engine.requests == 1)
        await managed.stop()
        #expect(await engine.released)
    }

    @Test("REF-CORE-02 A legacy service refuses a reference instead of silently dropping it")
    func legacy() async throws {
        let engine = FakeSpeechEngine()
        let live = LiveSpeechService(engine: engine, playback: AudioPlayback(), defaults: nil)
        let reference = try SpeechReferenceAudio(
            samples: [Float](repeating: 0.25, count: 24_000), source: .recording)
        await #expect(throws: SpeechError.referenceNotSupported) {
            try await live.generate(text: "Example", seed: 1, reference: reference)
        }
        #expect(await engine.requests.isEmpty)
    }
}

@MainActor
private final class ReferenceTestPermission {
    var allowed = true
}

private actor ReferenceServiceEngine: SpeechGenerating {
    var opened: URL?
    var reference: SpeechReferenceAudio?
    var requests = 0
    var released = false

    func open(_ directory: URL) -> SpeechModel {
        opened = directory
        return SpeechModel(name: "Reference", directory: directory, supportsReferenceAudio: true)
    }
    func generate(text: String, seed: UInt64) -> SpeechAudio {
        SpeechAudio(wav: Data(), audioSeconds: 1, elapsedSeconds: 1, normalizedText: text)
    }
    func generate(text: String, seed: UInt64, reference: SpeechReferenceAudio?) -> SpeechAudio {
        self.reference = reference
        requests += 1
        return generate(text: text, seed: seed)
    }
    func release() { released = true }
}
