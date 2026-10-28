#if os(macOS) || os(iOS)
    import Foundation
    import SpeechCore

    @MainActor
    public final class LiveSpeechService: SpeechService {
        public static let modelPathKey = "irodoriModelPath"
        private let engine: any SpeechGenerating
        private let playback: AudioPlayback
        private let defaults: UserDefaults?
        private let references: any SpeechReferenceReading
        private let recorder: any SpeechReferenceRecording

        public convenience init(defaults: UserDefaults? = .standard) {
            self.init(engine: IrodoriRuntime(), playback: AudioPlayback(), defaults: defaults)
        }
        init(
            engine: any SpeechGenerating, playback: AudioPlayback, defaults: UserDefaults?,
            references: any SpeechReferenceReading = SpeechReferenceReader(),
            recorder: any SpeechReferenceRecording = SpeechReferenceRecorder()
        ) {
            self.engine = engine
            self.playback = playback
            self.defaults = defaults
            self.references = references
            self.recorder = recorder
        }

        public func open(path: String) async throws -> SpeechModel {
            let model = try await engine.open(URL(filePath: path, directoryHint: .isDirectory))
            try Task.checkCancellation()
            defaults?.set(path, forKey: Self.modelPathKey)
            return model
        }
        public func generate(text: String, seed: UInt64) async throws -> SpeechAudio {
            try await engine.generate(text: text, seed: seed)
        }
        public func generate(text: String, seed: UInt64, reference: SpeechReferenceAudio?) async throws -> SpeechAudio {
            try await engine.generate(text: text, seed: seed, reference: reference)
        }
        public func play(_ data: Data) async throws { try await playback.play(data) }
        public func recordReference(ticket: Int, started: @MainActor () -> Void) async throws -> SpeechReferenceAudio {
            let url = try await recorder.record(ticket: ticket) { await started() }
            do {
                try Task.checkCancellation()
                let reference = try await references.read(url, source: .recording)
                await references.removeRecording(url)
                try Task.checkCancellation()
                return reference
            } catch {
                await references.removeRecording(url)
                throw error
            }
        }
        public func finishReferenceRecording(ticket: Int) async { await recorder.finish(ticket: ticket) }
        public func importReference(_ url: URL) async throws -> SpeechReferenceAudio {
            try await references.read(url, source: .file(url.lastPathComponent))
        }
        public func stop() async {
            await playback.stopAndWait()
            await recorder.stop()
            await engine.release()
        }
    }
#endif
