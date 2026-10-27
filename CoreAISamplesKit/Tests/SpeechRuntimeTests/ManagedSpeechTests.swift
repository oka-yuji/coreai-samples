import FeatureStore
import Foundation
import ModelLibraryFeature
import SpeechCore
import SpeechFeature
import Testing

@testable import SpeechRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ManagedSpeechTests {
    @Test("IR-SESSION-01 Binding a managed speech session does not open, generate or play a model")
    func lazyBinding() async {
        let fake = ManagedSpeechFake()
        let sessions = SpeechDemoSessions { fake }
        let session = session(1)
        let store = sessions.bind(session, examples: []) { true }
        #expect(fake.calls.isEmpty)
        #expect(store.state.managed && store.state.presented)
        store.send(.openBundle("/unmanaged"))
        #expect(fake.calls.isEmpty && store.state.model?.directory == session.resource.directory)
        await sessions.stop(session)
        #expect(fake.calls == ["stop"] || fake.calls == ["stop", "stop"])
    }

    @Test("IOS-IR-04 Closing a completed speech session keeps input but no result or status in its preview")
    func completedPreview() async throws {
        let fake = ManagedSpeechFake()
        let sessions = SpeechDemoSessions { fake }
        let session = session(1)
        let store = sessions.bind(session, examples: []) { true }
        store.send(.setText("Keep this input"))
        store.send(.setSeed("23"))
        store.send(.generate)
        for await state in Observations({ store.state }) where state.audio != nil { break }
        #expect(store.state.message == "Speech is ready. Play it or save the WAV file.")
        await sessions.stop(session)
        let preview = try #require(sessions.state(for: session.selection))
        #expect(preview.text == "Keep this input" && preview.seed == "23")
        #expect(preview.audio == nil && preview.message == nil && preview.model == nil)
        #expect(preview.exportTicket == nil && !preview.presented && preview.phase == .idle)
    }

    @Test("IR-SESSION-01 Only permitted speech operations open the selected resource and deliver results")
    func gateAndResults() async throws {
        let fake = ManagedSpeechFake()
        var permitted = false
        let service = ManagedSpeechService(service: fake, directory: URL(filePath: "/managed")) { permitted }
        let runner = SpeechRunner(service: service)
        var actions: [SpeechAction] = []
        await runner.run(.generate(1, "Test", 17)) { actions.append($0) }
        await runner.run(.play(2, Data([1]))) { actions.append($0) }
        #expect(fake.calls.isEmpty)
        #expect(!actions.contains { if case .generated = $0 { true } else { false } })
        actions.removeAll()
        permitted = true
        await runner.run(.generate(3, "Test", 17)) { actions.append($0) }
        await runner.run(.play(4, fake.audio.wav)) { actions.append($0) }
        #expect(fake.calls == ["open:/managed", "generate:Test:17", "play"])
        #expect(actions == [.generated(3, fake.audio), .playbackFinished(4)])
        await service.stop()
    }

    @Test("IR-SESSION-02 A managed speech failure and external playback interruption can be retried")
    func failures() async {
        let fake = ManagedSpeechFake()
        let service = ManagedSpeechService(service: fake, directory: URL(filePath: "/managed")) { true }
        let runner = SpeechRunner(service: service)
        var actions: [SpeechAction] = []
        fake.failGeneration = true
        await runner.run(.generate(1, "Test", 1)) { actions.append($0) }
        #expect(actions == [.failed(1, SpeechError.invalidSamples.localizedDescription)])
        fake.failGeneration = false
        await runner.run(.generate(2, "Test", 1)) { actions.append($0) }
        #expect(actions.last == .generated(2, fake.audio))
        fake.interruptPlayback = true
        await runner.run(.play(3, fake.audio.wav)) { actions.append($0) }
        #expect(actions.last == .playbackInterrupted(3))
        fake.interruptPlayback = false
        await runner.run(.play(4, fake.audio.wav)) { actions.append($0) }
        #expect(actions.last == .playbackFinished(4))
        await service.stop()
    }

    @Test("IOS-IR-04 Revoking permission while opening drops the result and joins the late resource before stopping")
    func stopDuringOpen() async throws {
        let fake = ManagedSpeechFake()
        fake.blockOpen = true
        defer { fake.openGate.open() }
        var permitted = true
        let service = ManagedSpeechService(service: fake, directory: URL(filePath: "/managed")) { permitted }
        let operation = Task { try await service.generate(text: "Test", seed: 2) }
        try await fake.openStarted.wait()
        permitted = false
        service.deactivate()
        operation.cancel()
        var stopped = false
        let stopping = Task {
            await service.stop()
            stopped = true
        }
        try await fake.stopStarted.wait()
        #expect(!stopped)
        fake.openGate.open()
        await #expect(throws: CancellationError.self) { try await operation.value }
        await stopping.value
        #expect(stopped && fake.calls == ["open:/managed", "stop", "stop"])
        await #expect(throws: CancellationError.self) { try await service.generate(text: "Again", seed: 3) }
    }

    @Test("IOS-IR-04 The managed Store drops late speech after closing and preserves only input for a new session")
    func sessionLifetime() async throws {
        let first = ManagedSpeechFake()
        first.blockGeneration = true
        first.blockStop = true
        let second = ManagedSpeechFake()
        var calls = 0
        let sessions = SpeechDemoSessions {
            calls += 1
            return calls == 1 ? first : second
        }
        defer {
            first.generateGate.open()
            first.stopGate.open()
        }
        let a = session(1)
        let old = sessions.bind(a, examples: []) { true }
        old.send(.setText("Retained text"))
        old.send(.setSeed("7"))
        old.send(.generate)
        try await first.generateStarted.wait()
        var stopped = false
        let stop = Task {
            await sessions.stop(a)
            stopped = true
        }
        try await first.stopStarted.wait()
        #expect(old.state.phase == .stopping && !old.state.presented && !stopped)
        old.send(.generate)
        first.generateGate.open()
        first.stopGate.open()
        await stop.value
        #expect(old.state.audio == nil && old.state.exportTicket == nil && stopped)
        let b = session(2)
        let current = sessions.bind(b, examples: []) { true }
        #expect(current.state.text == "Retained text" && current.state.seed == "7")
        #expect(current.state.audio == nil && current.state.phase == .idle && current.state.canGenerate)
        await sessions.stop(a)
        #expect(current.state.presented && second.calls.isEmpty)
        await sessions.stop(b)
    }

    @Test("IOS-IR-04 Stop joins playback and deactivation even when the stop caller is cancelled")
    func stopPlayback() async throws {
        let fake = ManagedSpeechFake()
        fake.blockPlayback = true
        fake.blockStop = true
        defer {
            fake.playGate.open()
            fake.stopGate.open()
        }
        let service = ManagedSpeechService(service: fake, directory: URL(filePath: "/managed")) { true }
        let playing = Task { try await service.play(fake.audio.wav) }
        try await fake.playStarted.wait()
        playing.cancel()
        var stopped = false
        let stopping = Task {
            await service.stop()
            stopped = true
        }
        try await fake.stopStarted.wait()
        stopping.cancel()
        #expect(!stopped)
        fake.playGate.open()
        await #expect(throws: CancellationError.self) { try await playing.value }
        #expect(!stopped)
        fake.stopGate.open()
        await stopping.value
        #expect(stopped)
        #expect(!fake.stopObservedCancellation)
        try await service.play(fake.audio.wav)
        await service.stop()
    }

    private func session(_ number: Int) -> ModelDemoSession {
        ModelDemoSession(
            id: LibraryTicket(
                owner: UUID(uuidString: "00000000-0000-0000-0000-000000000001") ?? UUID(), number: number),
            selection: DemoSelection(modelID: "irodori", demoID: "speech"),
            resource: ValidatedModelResource(
                modelID: "irodori", artifactID: "irodori-fp32", contract: "irodori-coreai-v1",
                revision: String(repeating: "a", count: 64), upstreamRevision: nil,
                directory: URL(filePath: "/managed"), sizeBytes: 1, sourceKind: .localManifest))
    }
}

@MainActor
private final class ManagedSpeechFake: SpeechService {
    let openStarted = TestSignal()
    let generateStarted = TestSignal()
    let playStarted = TestSignal()
    let stopStarted = TestSignal()
    let openGate = TestGate()
    let generateGate = TestGate()
    let playGate = TestGate()
    let stopGate = TestGate()
    var blockOpen = false
    var blockGeneration = false
    var blockPlayback = false
    var blockStop = false
    var failGeneration = false
    var interruptPlayback = false
    var stopObservedCancellation = false
    var calls: [String] = []
    let audio = SpeechAudio(wav: Data([1]), audioSeconds: 1, elapsedSeconds: 2, normalizedText: "Test")

    func open(path: String) async -> SpeechModel {
        calls.append("open:" + path)
        openStarted.fire()
        if blockOpen { await openGate.wait() }
        return SpeechModel(name: "Fixture", directory: URL(filePath: path))
    }
    func generate(text: String, seed: UInt64) async throws -> SpeechAudio {
        calls.append("generate:\(text):\(seed)")
        generateStarted.fire()
        if blockGeneration { await generateGate.wait() }
        if failGeneration { throw SpeechError.invalidSamples }
        return audio
    }
    func play(_ data: Data) async throws {
        calls.append("play")
        playStarted.fire()
        if blockPlayback { await playGate.wait() }
        if interruptPlayback { throw CancellationError() }
    }
    func stop() async {
        calls.append("stop")
        stopStarted.fire()
        if blockStop { await stopGate.wait() }
        stopObservedCancellation = stopObservedCancellation || Task.isCancelled
    }
}
