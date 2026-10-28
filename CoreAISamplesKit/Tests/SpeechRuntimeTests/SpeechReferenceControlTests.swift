import FeatureStore
import Foundation
import SpeechCore
import SpeechFeature
import Testing

@testable import SpeechRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct SpeechReferenceControlTests {
    @Test("REF-UI-02 Old finish requests cannot end a recording started after stop")
    func staleFinish() async throws {
        let service = try DelayedReferenceControlService()
        let runner = SpeechRunner(service: service)
        let stopped = TestSignal()
        let oldControlReturned = TestSignal()
        let newRecordingReturned = TestSignal()
        var initial = SpeechState(managedModel: service.open(path: "/fixture"))
        initial.presented = true
        let store = Store<SpeechFeature>(initialState: initial) { effect, send in
            await runner.run(effect, send: send)
            if case .stop = effect { stopped.fire() }
            if case .finishReferenceRecording(1) = effect { oldControlReturned.fire() }
            if case .recordReference(3) = effect { newRecordingReturned.fire() }
        }
        defer {
            service.allowFinish.open()
            service.releaseCurrent()
        }
        store.send(.recordReference)
        try await service.firstStarted.wait()
        store.send(.finishReferenceRecording)
        try await service.finishStarted.wait()
        store.send(.stop)
        try await stopped.wait()
        #expect(store.state.phase == .idle)
        store.send(.recordReference)
        try await service.secondStarted.wait()
        service.allowFinish.open()
        try await oldControlReturned.wait()
        #expect(store.state.phase == .recording && service.finished.isEmpty)
        await service.finishReferenceRecording(ticket: 1)
        #expect(service.finished.isEmpty)
        store.send(.finishReferenceRecording)
        try await newRecordingReturned.wait()
        #expect(service.finished == [3])
        #expect(store.state.phase == .idle && store.state.reference == service.voice)
    }
}

@MainActor
private final class DelayedReferenceControlService: SpeechService {
    let firstStarted = TestSignal()
    let secondStarted = TestSignal()
    let finishStarted = TestSignal()
    let allowFinish = TestGate()
    let voice: SpeechReferenceAudio
    var currentTicket: Int?
    var startedCount = 0
    var finished: [Int] = []
    var recording: CheckedContinuation<Void, Never>?

    init() throws {
        voice = try SpeechReferenceAudio(samples: [Float](repeating: 0.25, count: 24_000), source: .recording)
    }
    func open(path: String) -> SpeechModel {
        SpeechModel(name: "Reference", directory: URL(filePath: path), supportsReferenceAudio: true)
    }
    func generate(text: String, seed: UInt64) -> SpeechAudio {
        SpeechAudio(wav: Data(), audioSeconds: 1, elapsedSeconds: 1, normalizedText: text)
    }
    func play(_ data: Data) {}
    func recordReference(ticket: Int, started: @MainActor () -> Void) async throws -> SpeechReferenceAudio {
        currentTicket = ticket
        startedCount += 1
        started()
        await withCheckedContinuation { pending in
            recording = pending
            if startedCount == 1 { firstStarted.fire() } else { secondStarted.fire() }
        }
        return voice
    }
    func finishReferenceRecording(ticket: Int) async {
        if ticket == 1 {
            finishStarted.fire()
            await allowFinish.wait()
        }
        guard currentTicket == ticket, !Task.isCancelled else { return }
        finished.append(ticket)
        releaseCurrent()
    }
    func stop() { releaseCurrent() }
    func releaseCurrent() {
        currentTicket = nil
        let pending = recording
        recording = nil
        pending?.resume()
    }
}
