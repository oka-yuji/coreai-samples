import AVFAudio
import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import Observation
import SpeechFeature
import SpeechRuntime
import Testing

@testable import ModelLibraryRuntime

@MainActor
struct VoxCPMMacIntegrationTests {
    @Test(
        "The real bundle imports into Mac Models and every example generates through the speech store until the lease returns",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_VOXCPM2_MODELS_BUNDLE"] != nil),
        .timeLimit(.minutes(10)))
    func importedSpeech() async throws {
        let environment = ProcessInfo.processInfo.environment
        let source = URL(
            filePath: try #require(environment["COREAI_VOXCPM2_MODELS_BUNDLE"]), directoryHint: .isDirectory)
        let output = URL(
            filePath: try #require(environment["COREAI_VOXCPM2_MODELS_OUTPUT"]), directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        let resource = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "CoreAISamples/Demos/Speech/Resources/voxcpm2-speech-examples.json")
        let examples = try JSONDecoder().decode([SpeechExample].self, from: Data(contentsOf: resource))
        let root = FileManager.default.temporaryDirectory.appending(path: "voxcpm2-models-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "VoxCPMMacIntegrationTests." + UUID().uuidString
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let artifact = try VoxCPMModelArtifact.make()
        let clock = ContinuousClock()
        var seconds: [String: Double] = [:]
        var start = clock.now
        let before = try snapshot(source)
        seconds["sourceHash"] = (clock.now - start) / .seconds(1)
        #expect(
            before.filter { $0.value != FileAttributeType.typeDirectory.rawValue }
                == Dictionary(uniqueKeysWithValues: artifact.files.map { ($0.path, $0.sha256) }))

        let storage = ManagedModelStore(root: root)
        let assembly = ModelLibraryAssembly(adapters: [], store: storage, transfer: ArtifactTransfer())
        let effects = AsyncStream<ModelLibraryEffect>.makeStream()
        let library = Store<ModelLibraryFeature>(
            initialState: ModelLibraryState(
                catalog: try MacDemoCatalog.make(), environment: LibraryEnvironment(platform: .macOS), owner: UUID())
        ) { effect, send in
            await assembly.runner.run(effect, send: send)
            effects.continuation.yield(effect)
        }
        assembly.runner.register(library)
        var completed = effects.stream.makeAsyncIterator()
        start = clock.now
        library.send(.importRequested(artifact.id, source))
        while let effect = await completed.next() { if case .importBundle = effect { break } }
        seconds["import"] = (clock.now - start) / .seconds(1)
        let installed = try #require(library.state.installations[artifact.id]?.resource)
        #expect(library.state.installations[artifact.id]?.phase == .ready && installed.directory != source)
        var leases = ["afterImport": await storage.activeLeases(identity: installed.identity)]

        let speech = LiveSpeechService.makeStore(engine: .voxcpm2, examples: examples, defaults: defaults)
        speech.send(.appeared)
        #expect(!speech.state.canGenerate(usingModelAt: library.state.usages["voxcpm2"]?.resource?.directory.path))
        start = clock.now
        library.send(.useRequested("voxcpm2"))
        while let effect = await completed.next() { if case .acquireUsage = effect { break } }
        seconds["acquire"] = (clock.now - start) / .seconds(1)
        let used = try #require(library.state.usages["voxcpm2"]?.resource)
        #expect(used == installed)
        leases["inUse"] = await storage.activeLeases(identity: installed.identity)
        start = clock.now
        speech.send(.openBundle(used.directory.path))
        await settle(speech)
        seconds["open"] = (clock.now - start) / .seconds(1)
        let model = try #require(speech.state.model, "\(speech.state.message ?? "")")
        #expect(model.name == "VoxCPM2 (FP16)" && !model.supportsReferenceAudio)
        #expect(defaults.string(forKey: "voxcpm2ModelPath") == used.directory.path)
        #expect(defaults.string(forKey: "irodoriModelPath") == nil)
        #expect(speech.state.canGenerate(usingModelAt: used.directory.path))

        var generations: [[String: Any]] = []
        for example in examples {
            speech.send(.setText(example.text))
            start = clock.now
            speech.send(.generate)
            await settle(speech)
            let wall = (clock.now - start) / .seconds(1)
            let audio = try #require(speech.state.audio, "\(example.id): \(speech.state.message ?? "")")
            let wav = output.appending(path: example.id + ".wav")
            try audio.wav.write(to: wav, options: .withoutOverwriting)
            let file = try AVAudioFile(forReading: wav)
            #expect(file.fileFormat.sampleRate == 48_000 && file.fileFormat.channelCount == 1, "\(example.id)")
            #expect(file.fileFormat.commonFormat == .pcmFormatFloat32, "\(example.id)")
            #expect(file.length > 0 && file.length % 7680 == 0, "\(example.id)")
            #expect(Double(file.length) / 48_000 == audio.audioSeconds, "\(example.id)")
            generations.append([
                "id": example.id, "audioSeconds": audio.audioSeconds, "elapsedSeconds": audio.elapsedSeconds,
                "wallSeconds": wall, "samples": file.length, "patches": file.length / 7680, "wavBytes": audio.wav.count,
            ])
        }

        start = clock.now
        speech.send(.ended)
        await settle(speech)
        seconds["stop"] = (clock.now - start) / .seconds(1)
        leases["afterStop"] = await storage.activeLeases(identity: installed.identity)
        start = clock.now
        library.send(.useEnded("voxcpm2"))
        while let effect = await completed.next() { if case .releaseUsage = effect { break } }
        seconds["release"] = (clock.now - start) / .seconds(1)
        #expect(library.state.usages["voxcpm2"] == nil)
        leases["afterRelease"] = await storage.activeLeases(identity: installed.identity)
        #expect(leases == ["afterImport": 0, "inUse": 1, "afterStop": 1, "afterRelease": 0])
        #expect(try await storage.delete(identity: installed.identity) == installed.sizeBytes)
        start = clock.now
        let after = try snapshot(source)
        seconds["sourceRehash"] = (clock.now - start) / .seconds(1)
        #expect(after == before)
        let record: [String: Any] = [
            "format": "voxcpm2-models-integration/1", "system": ProcessInfo.processInfo.operatingSystemVersionString,
            "artifactID": artifact.id, "manifestSHA256": artifact.sourceRevision ?? "",
            "files": artifact.files.count, "bytes": installed.sizeBytes, "model": model.name, "seconds": seconds,
            "generations": generations, "leases": leases, "sourceUnchanged": after == before,
        ]
        try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appending(path: "voxcpm2-models.json"), options: .withoutOverwriting)
        effects.continuation.finish()
    }

    private func settle(_ store: Store<SpeechFeature>) async {
        for await phase in Observations({ store.state.phase }) where phase == .idle { return }
    }

    private func snapshot(_ folder: URL) throws -> [String: String] {
        var result: [String: String] = [:]
        for path in try FileManager.default.subpathsOfDirectory(atPath: folder.path) {
            let item = folder.appending(path: path)
            let type = try FileManager.default.attributesOfItem(atPath: item.path)[.type] as? FileAttributeType
            result[path] =
                type == .typeRegular ? sha256(try Data(contentsOf: item, options: .alwaysMapped)) : type?.rawValue ?? ""
        }
        return result
    }
}
