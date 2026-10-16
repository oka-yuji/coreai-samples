#if os(macOS)
    import Foundation
    import SpeechCore

    @MainActor
    public final class LiveSpeechService: SpeechService {
        public static let modelPathKey = "irodoriModelPath"
        private let engine: any SpeechGenerating
        private let playback: AudioPlayback
        private let defaults: UserDefaults

        public convenience init(defaults: UserDefaults = .standard) {
            self.init(engine: IrodoriRuntime(), playback: AudioPlayback(), defaults: defaults)
        }
        init(engine: any SpeechGenerating, playback: AudioPlayback, defaults: UserDefaults) {
            self.engine = engine
            self.playback = playback
            self.defaults = defaults
        }

        public func open(path: String) async throws -> SpeechModel {
            let model = try await engine.open(URL(filePath: path, directoryHint: .isDirectory))
            try Task.checkCancellation()
            defaults.set(path, forKey: Self.modelPathKey)
            return model
        }
        public func generate(text: String, seed: UInt64) async throws -> SpeechAudio {
            try await engine.generate(text: text, seed: seed)
        }
        public func play(_ data: Data) async throws { try await playback.play(data) }
        public func stop() async {
            playback.stop()
            await engine.release()
        }
    }
#endif
