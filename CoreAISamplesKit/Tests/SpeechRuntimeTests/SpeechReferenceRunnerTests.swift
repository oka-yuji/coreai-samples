import FeatureStore
import Foundation
import SpeechCore
import SpeechFeature
import Testing

@testable import SpeechRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct SpeechReferenceRunnerTests {
    @Test("REF-UI-01 Recording start finish and generation carry the selected reference through the runner")
    func recording() async throws {
        let reference = try voice()
        let service = ReferenceInputService(reference: reference)
        let runner = SpeechRunner(service: service)
        let completed = TestSignal()
        let store = Store<SpeechFeature>(initialState: ready()) { effect, send in
            await runner.run(effect, send: send)
            if case .recordReference = effect { completed.fire() }
        }
        defer {
            service.inputGate.open()
            service.stopGate.open()
        }
        store.send(.recordReference)
        try await service.began.wait()
        #expect(store.state.phase == .recording)
        store.send(.finishReferenceRecording)
        try await completed.wait()
        #expect(store.state.reference == reference && store.state.phase == .idle)
        var actions: [SpeechAction] = []
        await runner.run(.generateReference(8, "Example", 1708, reference)) { actions.append($0) }
        #expect(service.generatedReference == reference)
        #expect(actions == [.generated(8, service.audio)])
    }

    @Test("REF-UI-02 Ending during an uncooperative capture discards its result and awaits stop acknowledgement")
    func cancellation() async throws {
        let service = ReferenceInputService(reference: try voice())
        let runner = SpeechRunner(service: service)
        let captureReturned = TestSignal()
        let stopReturned = TestSignal()
        let store = Store<SpeechFeature>(initialState: ready()) { effect, send in
            await runner.run(effect, send: send)
            if case .recordReference = effect { captureReturned.fire() }
            if case .stop = effect { stopReturned.fire() }
        }
        defer {
            service.inputGate.open()
            service.stopGate.open()
        }
        store.send(.recordReference)
        try await service.began.wait()
        store.send(.ended)
        store.send(.appeared)
        try await service.stopping.wait()
        service.inputGate.open()
        try await captureReturned.wait()
        #expect(store.state.reference == nil && store.state.phase == .stopping)
        service.stopGate.open()
        try await stopReturned.wait()
        #expect(store.state.phase == .idle && store.state.canSelectReference)
    }

    @Test("REF-UI-01 The runner reports denied permission and file failures with their original session")
    func failures() async throws {
        let service = ReferenceInputService(reference: try voice())
        let runner = SpeechRunner(service: service)
        service.failure = .permissionDenied
        var actions: [SpeechAction] = []
        await runner.run(.recordReference(4)) { actions.append($0) }
        #expect(actions == [.microphoneDenied(4)])
        service.failure = .unreadable
        await runner.run(.importReference(5, URL(filePath: "/bad.wav"))) { actions.append($0) }
        #expect(actions.last == .failed(5, SpeechReferenceError.unreadable.localizedDescription))
        service.failure = nil
        await runner.run(.importReference(6, URL(filePath: "/voice.wav"))) { actions.append($0) }
        let expected = try SpeechReferenceAudio(samples: service.reference.samples, source: .file("voice.wav"))
        #expect(actions.last == .referencePrepared(6, expected))
    }

    private func voice() throws -> SpeechReferenceAudio {
        try SpeechReferenceAudio(samples: [Float](repeating: 0.25, count: 24_000), source: .recording)
    }

    private func ready() -> SpeechState {
        var state = SpeechState(
            managedModel: SpeechModel(
                name: "Reference", directory: URL(filePath: "/model"), supportsReferenceAudio: true))
        state.presented = true
        state.text = "Example"
        return state
    }
}

@MainActor
private final class ReferenceInputService: SpeechService {
    let reference: SpeechReferenceAudio
    let began = TestSignal()
    let inputGate = TestGate()
    let stopping = TestSignal()
    let stopGate = TestGate()
    var failure: SpeechReferenceError?
    var generatedReference: SpeechReferenceAudio?
    let audio = SpeechAudio(wav: Data([1]), audioSeconds: 1, elapsedSeconds: 1, normalizedText: "Example")
    init(reference: SpeechReferenceAudio) { self.reference = reference }
    func open(path: String) -> SpeechModel {
        SpeechModel(name: "Reference", directory: URL(filePath: path), supportsReferenceAudio: true)
    }
    func generate(text: String, seed: UInt64) -> SpeechAudio { audio }
    func generate(text: String, seed: UInt64, reference: SpeechReferenceAudio?) -> SpeechAudio {
        generatedReference = reference
        return audio
    }
    func recordReference(ticket: Int, started: @MainActor () -> Void) async throws -> SpeechReferenceAudio {
        if let failure { throw failure }
        started()
        began.fire()
        await inputGate.wait()
        return reference
    }
    func finishReferenceRecording(ticket: Int) { inputGate.open() }
    func importReference(_ url: URL) throws -> SpeechReferenceAudio {
        if let failure { throw failure }
        return try SpeechReferenceAudio(samples: reference.samples, source: .file(url.lastPathComponent))
    }
    func play(_ data: Data) {}
    func stop() async {
        stopping.fire()
        await stopGate.wait()
    }
}
