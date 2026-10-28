import Foundation
import SpeechCore
import SpeechFeature
import Testing

@testable import SpeechRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct PhoneSpeechCheckTests {
    let examples = [
        SpeechExample(id: "one", title: "One", text: "One"), SpeechExample(id: "two", title: "Two", text: "Two"),
    ]

    @Test("REF-UI-05 The device check uses its reference file and can save evidence without playing sound")
    func referenceCheck() async throws {
        let root = try phoneTestDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data([1]).write(to: root.appending(path: "reference.wav"))
        let service = FakeCheckSpeechService()
        let reference = try SpeechReferenceAudio(
            samples: [Float](repeating: 0.25, count: 24_000), source: .file("reference.wav"))
        service.referenceToImport = reference
        let check = PhoneSpeechCheck(
            service: service, files: SpeechCheckFiles(documents: root), examples: examples,
            environment: FakePhoneEnvironment())
        let result = try await check.run(
            .init(folder: "model", runID: "reference", referenceAudio: "reference.wav", playAudio: false)
        ) { _ in }
        let report = try JSONDecoder().decode(
            PhoneSpeechReport.self, from: Data(contentsOf: root.appending(path: result).appending(path: "report.json")))
        #expect(service.generatedReferences == [reference, reference, reference])
        #expect(service.plays == 0 && !report.playbackCompleted)
        #expect(report.referenceAudioSeconds == 0.5 && report.repeatedWAVMatches == true)
    }

    @Test("The check saves three independent samples, playback evidence and resource snapshots")
    func successfulCheck() async throws {
        let root = try phoneTestDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let live = FakeCheckSpeechService()
        let environment = FakePhoneEnvironment()
        let check = PhoneSpeechCheck(
            service: live, files: SpeechCheckFiles(documents: root), examples: examples, environment: environment)
        let result = try await check.run(.init(folder: "model", runID: "first")) { _ in }
        let report = try JSONDecoder().decode(
            PhoneSpeechReport.self,
            from: Data(contentsOf: root.appendingPathComponent(result).appendingPathComponent("report.json")))
        #expect(report.records.count == 3 && report.playbackCompleted)
        #expect(report.startMemory.availableBytes == 100 && report.finalMemory.footprintBytes == 20)
        #expect(live.texts == ["One", "Two", "One"] && live.seeds == [1708, 1708, 1708])
        #expect(live.plays == 1 && live.stops == 1)
        #expect(environment.begins == 1 && environment.ends == 1)
        for index in 0..<3 {
            #expect(
                try Data(contentsOf: root.appendingPathComponent(result).appendingPathComponent("sample-\(index).wav"))
                    == Data([1, 2]))
        }
    }

    @Test(
        "Open, generation and playback failures release resources and write failure status",
        arguments: ["open", "generate", "play"])
    func failures(stage: String) async throws {
        let root = try phoneTestDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let live = FakeCheckSpeechService()
        live.failure = stage
        let environment = FakePhoneEnvironment()
        let check = PhoneSpeechCheck(
            service: live, files: SpeechCheckFiles(documents: root), examples: examples, environment: environment)
        await #expect(throws: SpeechError.invalidSamples) {
            try await check.run(.init(folder: "model", runID: "failed")) { _ in }
        }
        let status = try JSONDecoder().decode(
            [String: String].self,
            from: Data(contentsOf: root.appendingPathComponent("SpeechChecks/failed/status.json")))
        #expect(status["phase"] == "failed")
        #expect(live.stops == 1 && environment.ends == 1)
        #expect(
            !FileManager.default.fileExists(atPath: root.appendingPathComponent("SpeechChecks/failed/report.json").path)
        )
    }

    @Test("Missing check examples fail before loading a model and restore the environment")
    func missingExamples() async throws {
        let root = try phoneTestDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let live = FakeCheckSpeechService()
        let environment = FakePhoneEnvironment()
        let check = PhoneSpeechCheck(
            service: live, files: SpeechCheckFiles(documents: root), examples: [], environment: environment)
        await #expect(throws: (any Error).self) { try await check.run(.init(folder: "model", runID: "empty")) { _ in } }
        #expect(live.opens == 0 && environment.ends == 1)
    }

    @Test("Stop waits for a cancelled check to finish cleanup and refuses a new check meanwhile")
    func stopWaits() async throws {
        let root = try phoneTestDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let live = FakeCheckSpeechService()
        live.block = true
        let environment = FakePhoneEnvironment()
        let check = PhoneSpeechCheck(
            service: live, files: SpeechCheckFiles(documents: root), examples: examples, environment: environment)
        let running = Task { try await check.run(.init(folder: "model", runID: "cancelled")) { _ in } }
        defer {
            live.gate.open()
            running.cancel()
        }
        try await live.started.wait()
        running.cancel()
        var stopped = false
        let stopping = Task {
            await check.stop()
            stopped = true
        }
        try await live.stopCalled.wait()
        #expect(!stopped && environment.ends == 0)
        await #expect(throws: SpeechError.busy) {
            try await check.run(.init(folder: "model", runID: "overlap")) { _ in }
        }
        live.gate.open()
        await #expect(throws: CancellationError.self) { try await running.value }
        await stopping.value
        #expect(stopped && environment.ends == 1)
    }

    @Test("An external audio interruption ends the diagnostic instead of leaving it running")
    func interruptionEndsRunner() async throws {
        let root = try phoneTestDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let live = FakeCheckSpeechService()
        live.failure = "interrupt"
        let check = PhoneSpeechCheck(
            service: live, files: SpeechCheckFiles(documents: root), examples: examples,
            environment: FakePhoneEnvironment())
        var actions: [SpeechCheckAction] = []
        await SpeechCheckRunner(service: check).run(.run(8, .init(folder: "model", runID: "interrupted"))) {
            actions.append($0)
        }
        #expect(actions.last == .failed(8, "The device check was interrupted."))
    }
}
