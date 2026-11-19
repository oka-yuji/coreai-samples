import Foundation
import SpeechCore
import SpeechRuntime

@MainActor
final class FakeVoiceTranslationService: SpeechService {
    let started = TestSignal()
    let gate = TestGate()
    let translation: String?
    let audio = SpeechAudio(wav: Data([1]), audioSeconds: 1, elapsedSeconds: 2, normalizedText: "Test")
    private(set) var voices: [String] = []
    private(set) var texts: [String] = []
    private(set) var seeds: [UInt64] = []

    init(translation: String?) { self.translation = translation }

    func open(path: String) -> SpeechModel { SpeechModel(name: "Fixture", directory: URL(filePath: path)) }
    func translate(voice: String) async throws -> String {
        voices.append(voice)
        started.fire()
        await gate.wait()
        guard let translation else { throw SpeechError.voiceNotTranslated }
        return translation
    }
    func generate(text: String, seed: UInt64) async throws -> SpeechAudio {
        texts.append(text)
        seeds.append(seed)
        started.fire()
        await gate.wait()
        return audio
    }
    func play(_ data: Data) {}
    func stop() async {}
}
