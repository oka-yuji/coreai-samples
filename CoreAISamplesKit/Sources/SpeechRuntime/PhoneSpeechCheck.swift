#if os(macOS) || os(iOS)
    import Foundation
    import SpeechCore
    import SpeechFeature

    struct PhoneMemory: Codable, Sendable {
        let availableBytes: UInt64
        let footprintBytes: UInt64?
        let peakFootprintBytes: Int64?
        let thermalState: Int
    }
    struct PhoneDeviceInfo: Sendable {
        let osVersion: String
        let osBuild: String
        let hardware: String
        let batteryState: Int
        let lowPowerMode: Bool
        let architecture: String
    }
    @MainActor
    protocol PhoneCheckEnvironment {
        func begin()
        func end()
        func memory() -> PhoneMemory
        func info() -> PhoneDeviceInfo
    }

    struct PhoneSpeechRecord: Codable, Sendable {
        let exampleID: String
        let seed: UInt64
        let normalizedText: String
        let audioSeconds: Double
        let generationSeconds: Double
        let wavBytes: Int
        let memory: PhoneMemory
    }
    struct PhoneSpeechReport: Codable, Sendable {
        let osVersion: String
        let osBuild: String
        let hardware: String
        let batteryState: Int
        let lowPowerMode: Bool
        let architecture: String
        let startMemory: PhoneMemory
        let records: [PhoneSpeechRecord]
        let playbackCompleted: Bool
        let finalMemory: PhoneMemory
    }

    @MainActor
    public final class PhoneSpeechCheck: SpeechDeviceChecking {
        private let service: any SpeechService
        private let files: SpeechCheckFiles
        private let examples: [SpeechExample]
        private var running = false
        private var stopping = 0
        private let environment: any PhoneCheckEnvironment
        private var stopWaiters: [CheckedContinuation<Void, Never>] = []
        init(
            service: any SpeechService, files: SpeechCheckFiles, examples: [SpeechExample],
            environment: any PhoneCheckEnvironment
        ) {
            self.service = service
            self.files = files
            self.examples = examples
            self.environment = environment
        }
        #if os(iOS)
            public convenience init(documents: URL, examples: [SpeechExample]) {
                self.init(
                    service: LiveSpeechService(), files: SpeechCheckFiles(documents: documents),
                    examples: examples, environment: LivePhoneCheckEnvironment())
            }
        #endif
        public func run(_ request: SpeechCheckRequest, progress: @MainActor (String) -> Void) async throws
            -> String
        {
            guard !running, stopping == 0 else { throw SpeechError.busy }
            running = true
            defer {
                running = false
                let waiters = stopWaiters
                stopWaiters.removeAll()
                for waiter in waiters { waiter.resume() }
            }
            let destination = try await files.reserve(request.runID)
            environment.begin()
            defer { environment.end() }
            let initial = environment.memory()
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            do {
                guard examples.count >= 2 else {
                    throw SpeechError.invalidAsset("The device-check examples are missing.")
                }
                let folder = try await files.modelFolder(request.folder)
                progress("Checking all model files…")
                try await files.writeStatus(encoder.encode(["phase": "opening", "os": environment.info().osVersion]))
                _ = try await service.open(path: folder.path)
                var records: [PhoneSpeechRecord] = []
                var audioToPlay: Data?
                for (index, example) in [examples[0], examples[1], examples[0]].enumerated() {
                    try Task.checkCancellation()
                    progress("Generating sample \(index + 1) of 3…")
                    try await files.writeStatus(encoder.encode(["phase": "generating", "sample": String(index)]))
                    let audio = try await service.generate(text: example.text, seed: 1708)
                    try Task.checkCancellation()
                    try await files.writeWAV(audio.wav, index: index)
                    records.append(
                        PhoneSpeechRecord(
                            exampleID: example.id, seed: 1708,
                            normalizedText: audio.normalizedText, audioSeconds: audio.audioSeconds,
                            generationSeconds: audio.elapsedSeconds, wavBytes: audio.wav.count,
                            memory: environment.memory()))
                    if index == 0 { audioToPlay = audio.wav }
                }
                guard let audioToPlay else { throw SpeechError.invalidSamples }
                try Task.checkCancellation()
                progress("Playing the first generated sample…")
                try await files.writeStatus(encoder.encode(["phase": "playing"]))
                try await service.play(audioToPlay)
                await service.stop()
                try Task.checkCancellation()
                let info = environment.info()
                let report = PhoneSpeechReport(
                    osVersion: info.osVersion, osBuild: info.osBuild,
                    hardware: info.hardware, batteryState: info.batteryState, lowPowerMode: info.lowPowerMode,
                    architecture: info.architecture, startMemory: initial, records: records,
                    playbackCompleted: true, finalMemory: environment.memory())
                try await files.writeReport(encoder.encode(report))
                try await files.writeStatus(encoder.encode(["phase": "completed"]))
                return "SpeechChecks/" + destination.lastPathComponent
            } catch {
                await service.stop()
                try? await files.writeStatus(encoder.encode(["phase": "failed", "error": error.localizedDescription]))
                throw error
            }
        }
        public func stop() async {
            stopping += 1
            defer { stopping -= 1 }
            await service.stop()
            if running { await withCheckedContinuation { stopWaiters.append($0) } }
        }
    }
#endif
