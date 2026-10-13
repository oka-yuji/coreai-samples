import FeatureStore
import Foundation
import SpeechCore
import SpeechFeature
import Testing

@testable import SpeechRuntime

actor FakeSpeechEngine: SpeechGenerating {
    let releaseStarted = TestSignal()
    let releaseGate = TestGate()
    var failOpen = false
    var released = false
    var requests: [String] = []
    func configure(failOpen: Bool) { self.failOpen = failOpen }
    func open(_ directory: URL) throws -> SpeechModel {
        if failOpen { throw SpeechError.invalidAsset("fixture") }
        return SpeechModel(name: "Fixture", directory: directory)
    }
    func generate(text: String, seed: UInt64) -> SpeechAudio {
        requests.append("\(text):\(seed)")
        return SpeechAudio(wav: Data([1]), audioSeconds: 1, elapsedSeconds: 2, normalizedText: text)
    }
    func release() async {
        releaseStarted.fire()
        await releaseGate.wait()
        released = true
    }
}

@MainActor
final class FakeSpeechService: SpeechService {
    let started = TestSignal()
    let gate = TestGate()
    let stopping = TestSignal()
    let stopGate = TestGate()
    var shouldFail = false
    var interruptPlayback = false
    let audio = SpeechAudio(wav: Data([1]), audioSeconds: 1, elapsedSeconds: 2, normalizedText: "Test")
    func open(path: String) -> SpeechModel { SpeechModel(name: "Fixture", directory: URL(filePath: path)) }
    func generate(text: String, seed: UInt64) async throws -> SpeechAudio {
        started.fire()
        await gate.wait()
        if shouldFail { throw SpeechError.invalidSamples }
        return audio
    }
    func play(_ data: Data) throws {
        if interruptPlayback { throw CancellationError() }
        if shouldFail { throw SpeechError.playbackFailed }
    }
    func stop() async {
        stopping.fire()
        await stopGate.wait()
    }
}

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct SpeechServiceTests {
    @Test("Only successful model validation persists a path and generation forwards text and seed")
    func persistenceAndInput() async throws {
        let name = "SpeechServiceTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let engine = FakeSpeechEngine()
        let service = LiveSpeechService(engine: engine, playback: AudioPlayback(), defaults: defaults)
        let model = try await service.open(path: "/good")
        #expect(model.directory.path == "/good")
        #expect(defaults.string(forKey: LiveSpeechService.modelPathKey) == "/good")
        await engine.configure(failOpen: true)
        await #expect(throws: SpeechError.invalidAsset("fixture")) { try await service.open(path: "/bad") }
        #expect(defaults.string(forKey: LiveSpeechService.modelPathKey) == "/good")
        let result = try await service.generate(text: "Test input", seed: 1708)
        #expect(result.normalizedText == "Test input")
        #expect(await engine.requests == ["Test input:1708"])
    }

    @Test("Service stop ends playback immediately and awaits runtime release")
    func stopRelease() async throws {
        let engine = FakeSpeechEngine()
        let device = FakePlaybackDevice()
        let playback = AudioPlayback { _ in device }
        let service = LiveSpeechService(engine: engine, playback: playback, defaults: .standard)
        let playing = Task { try await service.play(Data()) }
        defer {
            playback.stop()
            engine.releaseGate.open()
            playing.cancel()
        }
        try await device.started.wait()
        let stopping = Task { await service.stop() }
        try await engine.releaseStarted.wait()
        #expect(device.stopCount == 1)
        #expect(await engine.released == false)
        await #expect(throws: CancellationError.self) { try await playing.value }
        engine.releaseGate.open()
        await stopping.value
        #expect(await engine.released)
    }

    @Test("Runner reports outcomes with their session and suppresses cancelled results")
    func runnerOutcomes() async throws {
        let service = FakeSpeechService()
        service.gate.open()
        let runner = SpeechRunner(service: service)
        var actions: [SpeechAction] = []
        await runner.run(.open(10, "/model")) { actions.append($0) }
        await runner.run(.generate(11, "Test", 3)) { actions.append($0) }
        await runner.run(.play(12, Data())) { actions.append($0) }
        #expect(
            actions == [
                .opened(10, SpeechModel(name: "Fixture", directory: URL(filePath: "/model"))),
                .generated(11, service.audio), .playbackFinished(12),
            ])
        service.shouldFail = true
        await runner.run(.play(13, Data())) { actions.append($0) }
        #expect(actions.last == .failed(13, SpeechError.playbackFailed.localizedDescription))
        let gated = FakeSpeechService()
        let cancelledRunner = SpeechRunner(service: gated)
        let task = Task { await cancelledRunner.run(.generate(14, "Test", 3)) { actions.append($0) } }
        defer {
            gated.gate.open()
            task.cancel()
        }
        try await gated.started.wait()
        task.cancel()
        gated.gate.open()
        await task.value
        #expect(actions.count == 4)
    }

    @Test("Cancelled stop finishes cleanup without delivering an obsolete acknowledgement")
    func cancelledStop() async throws {
        let service = FakeSpeechService()
        let runner = SpeechRunner(service: service)
        var actions: [SpeechAction] = []
        var completed = false
        let task = Task {
            await runner.run(.stop(9)) { actions.append($0) }
            completed = true
        }
        defer {
            service.stopGate.open()
            task.cancel()
        }
        try await service.stopping.wait()
        task.cancel()
        #expect(!completed)
        service.stopGate.open()
        await task.value
        #expect(completed && actions.isEmpty)
    }

    @Test("Store cancellation ignores a late generation and keeps restart blocked until stop completes")
    func storeLifetime() async throws {
        let service = FakeSpeechService()
        let runner = SpeechRunner(service: service)
        var state = SpeechState()
        state.presented = true
        state.text = "Test"
        state.model = SpeechModel(name: "Fixture", directory: URL(filePath: "/model"))
        let stopped = TestSignal()
        let store = Store<SpeechFeature>(initialState: state) { effect, send in
            await runner.run(effect, send: send)
            if case .stop = effect { stopped.fire() }
        }
        defer {
            service.gate.open()
            service.stopGate.open()
        }
        store.send(.generate)
        try await service.started.wait()
        store.send(.ended)
        store.send(.appeared)
        try await service.stopping.wait()
        service.gate.open()
        #expect(store.state.phase == .stopping && !store.state.canGenerate)
        service.stopGate.open()
        try await stopped.wait()
        #expect(store.state.phase == .idle && store.state.canGenerate && store.state.audio == nil)
    }
}
