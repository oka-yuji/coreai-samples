import AVFAudio
import FeatureStore
import Foundation
import ModelLibraryFeature
import ModelLibraryStorage
import ModelLibraryTransfer
import Observation
import SpeechFeature
import SpeechRuntime
import Synchronization
import Testing

@testable import ModelLibraryRuntime

@MainActor
struct VoxCPMDownloadNativeTests {
    @Test(
        "VDL-04 A fresh download of the pinned release becomes ready and generates speech until the lease returns",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_VOXCPM2_DOWNLOAD_OUTPUT"] != nil),
        .timeLimit(.minutes(60)))
    func freshDownload() async throws {
        let output = URL(
            filePath: try #require(ProcessInfo.processInfo.environment["COREAI_VOXCPM2_DOWNLOAD_OUTPUT"]),
            directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        let resource = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "CoreAISamples/Demos/Speech/Resources/voxcpm2-speech-examples.json")
        let example = try #require(
            try JSONDecoder().decode([SpeechExample].self, from: Data(contentsOf: resource)).first)
        let suite = "VoxCPMDownloadNativeTests." + UUID().uuidString
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let catalog = try MacDemoCatalog.make()
        let model = try #require(catalog.model("voxcpm2"))
        let artifact = try #require(model.artifacts.first)
        let source = try #require(artifact.download)
        try #require(artifact == (try VoxCPMModelArtifact.makeDownload()))
        try #require(
            source.repository == "okayuji/voxcpm2-coreai"
                && source.revision == "afe11d61de9dd7c3609140aa040b777ef77b2555" && source.root == "fp16")
        try #require(artifact.files.count == 22 && artifact.totalFileBytes == 4_677_413_664)
        let prefix = "https://huggingface.co/\(source.repository)/resolve/\(source.revision)/\(source.root)/"
        let expected = Set(artifact.files.map { prefix + $0.path })
        let responses = Mutex<[String: Int]>([:])
        let transfer = ArtifactTransfer(fileLoader: { request, destination in
            let address = try #require(request.url?.absoluteString)
            try #require(expected.contains(address) && (request.allHTTPHeaderFields ?? [:]).isEmpty)
            let response = try await ArtifactTransfer.load(request, to: destination)
            let status = try #require((response as? HTTPURLResponse)?.statusCode)
            responses.withLock { $0[address] = status }
            return response
        })
        let root = FileManager.default.temporaryDirectory.appending(path: "voxcpm2-download-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try #require(!FileManager.default.fileExists(atPath: root.path))
        let storage = ManagedModelStore(root: root)
        guard case .missing = try await storage.inspect(model: model, artifact: artifact) else {
            Issue.record("The new managed area already holds VoxCPM2")
            return
        }
        let assembly = ModelLibraryAssembly(
            adapters: [], store: storage, transfer: transfer, migrationCandidates: { [] })
        let effects = AsyncStream<ModelLibraryEffect>.makeStream()
        defer { effects.continuation.finish() }
        let library = Store<ModelLibraryFeature>(
            initialState: ModelLibraryState(
                catalog: catalog, environment: LibraryEnvironment(platform: .macOS), owner: UUID())
        ) { effect, send in
            await assembly.runner.run(effect, send: send)
            effects.continuation.yield(effect)
        }
        assembly.runner.register(library)
        var completed = effects.stream.makeAsyncIterator()
        let clock = ContinuousClock()
        var seconds: [String: Double] = [:]

        var start = clock.now
        library.send(.downloadRequested(artifact.id))
        while let effect = await completed.next() { if case .download = effect { break } }
        seconds["download"] = (clock.now - start) / .seconds(1)
        try #require(
            library.state.installations[artifact.id]?.phase == .validating,
            "\(String(describing: library.state.installations[artifact.id]?.failure))")
        start = clock.now
        while let effect = await completed.next() { if case .validate = effect { break } }
        seconds["validation"] = (clock.now - start) / .seconds(1)
        try #require(library.state.installations[artifact.id]?.phase == .ready)
        let installed = try #require(library.state.installations[artifact.id]?.resource)
        let requests = responses.withLock { $0 }
        #expect(Set(requests.keys) == expected && requests.values.allSatisfy { $0 == 200 })
        #expect(installed.sourceKind == .localManifest && installed.revision == artifact.sourceRevision)
        #expect(installed.sizeBytes == 4_677_413_664 && installed.directory.path.hasPrefix(root.path + "/"))
        start = clock.now
        guard
            case .ready(let verified) = try await storage.inspect(
                model: model, artifact: artifact, verification: .full)
        else {
            Issue.record("The downloaded bundle did not pass the full inspection")
            return
        }
        seconds["fullInspection"] = (clock.now - start) / .seconds(1)
        #expect(verified == installed)
        let managedPath = installed.directory.pathComponents.dropFirst(root.pathComponents.count)
        try JSONSerialization.data(
            withJSONObject: [
                "format": "voxcpm2-download-acquisition/1", "repository": source.repository,
                "revision": source.revision, "root": source.root, "requests": requests,
                "requestsWithHeaders": 0, "files": artifact.files.count, "bytes": installed.sizeBytes,
                "sourceKind": installed.sourceKind.rawValue, "manifestSHA256": installed.revision,
                "managedPath": managedPath.joined(separator: "/"),
                "freshManagedStore": true, "migrationCandidates": 0, "seconds": seconds,
            ], options: [.prettyPrinted, .sortedKeys]
        ).write(to: output.appending(path: "acquisition.json"), options: .withoutOverwriting)
        var leases = ["afterDownload": await storage.activeLeases(identity: installed.identity)]

        let speech = LiveSpeechService.makeStore(engine: .voxcpm2, examples: [example], defaults: defaults)
        speech.send(.appeared)
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
        let opened = try #require(speech.state.model, "\(speech.state.message ?? "")")
        #expect(opened.name == "VoxCPM2 (FP16)" && speech.state.canGenerate(usingModelAt: used.directory.path))
        speech.send(.setText(example.text))
        start = clock.now
        speech.send(.generate)
        await settle(speech)
        let wall = (clock.now - start) / .seconds(1)
        let audio = try #require(speech.state.audio, "\(example.id): \(speech.state.message ?? "")")
        let wav = output.appending(path: example.id + ".wav")
        try audio.wav.write(to: wav, options: .withoutOverwriting)
        let file = try AVAudioFile(forReading: wav)
        #expect(file.fileFormat.sampleRate == 48_000 && file.fileFormat.channelCount == 1)
        #expect(file.fileFormat.commonFormat == .pcmFormatFloat32)
        #expect(file.length > 0 && file.length % 7680 == 0 && Double(file.length) / 48_000 == audio.audioSeconds)

        start = clock.now
        speech.send(.ended)
        await settle(speech)
        seconds["stop"] = (clock.now - start) / .seconds(1)
        leases["afterStop"] = await storage.activeLeases(identity: installed.identity)
        start = clock.now
        library.send(.useEnded("voxcpm2"))
        while let effect = await completed.next() { if case .releaseUsage = effect { break } }
        seconds["release"] = (clock.now - start) / .seconds(1)
        leases["afterRelease"] = await storage.activeLeases(identity: installed.identity)
        #expect(library.state.usages["voxcpm2"] == nil)
        #expect(leases == ["afterDownload": 0, "inUse": 1, "afterStop": 1, "afterRelease": 0])
        start = clock.now
        #expect(try await storage.delete(identity: installed.identity) == installed.sizeBytes)
        seconds["delete"] = (clock.now - start) / .seconds(1)
        try JSONSerialization.data(
            withJSONObject: [
                "format": "voxcpm2-download/1", "system": ProcessInfo.processInfo.operatingSystemVersionString,
                "repository": source.repository, "revision": source.revision, "root": source.root,
                "artifactID": artifact.id, "manifestSHA256": installed.revision, "files": artifact.files.count,
                "bytes": installed.sizeBytes, "model": opened.name, "seconds": seconds, "leases": leases,
                "generation": [
                    "id": example.id, "audioSeconds": audio.audioSeconds, "elapsedSeconds": audio.elapsedSeconds,
                    "wallSeconds": wall, "samples": file.length, "patches": file.length / 7680,
                    "wavBytes": audio.wav.count,
                ],
            ], options: [.prettyPrinted, .sortedKeys]
        ).write(to: output.appending(path: "voxcpm2-download.json"), options: .withoutOverwriting)
    }

    private func settle(_ store: Store<SpeechFeature>) async {
        for await phase in Observations({ store.state.phase }) where phase == .idle { return }
    }
}
