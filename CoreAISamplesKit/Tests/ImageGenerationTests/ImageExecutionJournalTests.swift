import Foundation
import ImageGenerationCore
import Testing

@testable import ImageGenerationRuntime

@Suite(.timeLimit(.minutes(1)))
struct ImageExecutionJournalTests {
    @Test("A persisted checkpoint precedes model loading and cancellation while recording prevents that load")
    func checkpointBeforeLoad() async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let backend = ImageFakeBackend()
        let (gate, resume) = AsyncStream<Void>.makeStream()
        defer { resume.finish() }
        let journal = HeldImageJournal(directory: directory, phase: .loading(.text), gate: gate)
        var events = journal.events.makeAsyncIterator()
        let pipeline = fixture.pipeline(backend: backend, journal: journal)
        let request = try fixture.request()
        let running = Task { try await pipeline.generate(request, progress: { _ in }) }
        while let phase = await events.next(isolation: #isolation), phase != .loading(.text) {}
        let saved = try checkpoint(directory)
        let report = try #require(saved["report"] as? [String: Any])
        #expect(saved["phase"] as? String == "loading")
        #expect(saved["component"] as? String == "text")
        #expect(report["requestID"] as? String == request.id.uuidString)
        #expect(report["manifestSHA256"] as? String == fixture.budget.manifestSHA256)
        #expect(report["status"] as? String == "running")
        #expect(saved["processID"] as? Int32 == ProcessInfo.processInfo.processIdentifier)
        #expect(await backend.history.isEmpty)
        running.cancel()
        resume.finish()
        await #expect(throws: CancellationError.self) { try await running.value }
        try await pipeline.stop()
        #expect(await backend.history.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: directory.appending(path: "completion.json").path))
        #expect(try checkpoint(directory)["sequence"] as? Int == 2)
    }

    @Test("A recording failure after load prevents inference and still releases the loaded resource")
    func recordingFailureReleases() async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let backend = ImageFakeBackend()
        let journal = HeldImageJournal(directory: directory, failing: .running(.text, step: nil))
        let pipeline = fixture.pipeline(backend: backend, journal: journal)
        await #expect(throws: ImageTestError.native) {
            try await pipeline.generate(fixture.request(), progress: { _ in })
        }
        #expect(await backend.history == ["load:text", "unload:text"])
        #expect(await pipeline.report?.status == .failed)
        #expect(try checkpoint(directory)["phase"] as? String == "loading")
        try await pipeline.stop()
    }

    @Test("A completed generation keeps its last checkpoint distinct from a successful terminal report")
    func completeRun() async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let journal = ImageExecutionJournal(directory: directory)
        let pipeline = fixture.pipeline(backend: ImageFakeBackend(), journal: journal)
        let result = try await pipeline.generate(fixture.request(), progress: { _ in })
        let saved = try checkpoint(directory)
        let report = try #require(saved["report"] as? [String: Any])
        #expect(result.report.status == .completed)
        #expect(result.report.diagnosticsEnabled)
        #expect(saved["phase"] as? String == "release")
        #expect(saved["component"] as? String == "vae_decoder")
        #expect(report["status"] as? String == "running")
        let completed = try #require(report["phases"] as? [[String: Any]])
        #expect(completed.count + 1 == result.report.phases.count)
    }

    @Test("Another request or journal cannot replace a checkpoint and unavailable memory remains absent")
    func identityAndUnavailableMemory() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let journal = ImageExecutionJournal(directory: directory)
        let identifier = UUID()
        let progress = ImageExecutionProgress(requestID: identifier, phase: .preparing)
        let report = ImageExecutionReport(
            requestID: identifier, profileID: "fixture", manifestSHA256: String(repeating: "a", count: 64),
            noiseSource: "fixture", diagnosticsEnabled: true)
        let memory = ImageMemoryObservation(
            scope: .unavailable, availableBytes: nil, footprintBytes: nil, physicalMemoryBytes: 1)
        try await journal.record(progress, report: report, memory: memory)
        let original = try Data(contentsOf: directory.appending(path: "checkpoint.json"))
        let savedMemory = try #require(try checkpoint(directory)["memory"] as? [String: Any])
        #expect(savedMemory["availableBytes"] == nil)
        #expect(savedMemory["footprintBytes"] == nil)
        await #expect(throws: ImageExecutionError.invalidResource) {
            try await journal.record(
                ImageExecutionProgress(requestID: UUID(), phase: .preparing), report: report, memory: memory)
        }
        await #expect(throws: ImageExecutionError.invalidResource) {
            try await ImageExecutionJournal(directory: directory).record(progress, report: report, memory: memory)
        }
        #expect(try Data(contentsOf: directory.appending(path: "checkpoint.json")) == original)
    }

    @Test("A failed atomic replacement preserves the last complete checkpoint and its sequence")
    func failedAtomicReplacement() async throws {
        let directory = try temporaryDirectory()
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
            try? FileManager.default.removeItem(at: directory)
        }
        let journal = ImageExecutionJournal(directory: directory)
        let identifier = UUID()
        let report = ImageExecutionReport(
            requestID: identifier, profileID: "fixture", manifestSHA256: String(repeating: "a", count: 64),
            noiseSource: "fixture", diagnosticsEnabled: true)
        let memory = ImageTestMemory().read()
        try await journal.record(
            ImageExecutionProgress(requestID: identifier, phase: .preparing), report: report, memory: memory)
        let original = try Data(contentsOf: directory.appending(path: "checkpoint.json"))
        try FileManager.default.setAttributes([.posixPermissions: 0o500], ofItemAtPath: directory.path)
        await #expect(throws: (any Error).self) {
            try await journal.record(
                ImageExecutionProgress(requestID: identifier, phase: .loading(.text)), report: report, memory: memory)
        }
        #expect(try Data(contentsOf: directory.appending(path: "checkpoint.json")) == original)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        try await journal.record(
            ImageExecutionProgress(requestID: identifier, phase: .loading(.text)), report: report, memory: memory)
        #expect(try checkpoint(directory)["sequence"] as? Int == 2)
    }

    @Test("Native allocation checkpoints require a request and follow its current component and step")
    func nativeCheckpointBinding() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let journal = ImageExecutionJournal(directory: directory)
        let memory = ImageTestMemory().read()
        await #expect(throws: ImageExecutionError.invalidResource) {
            try await journal.recordNative(stage: "run-before", asset: "text/text.aimodel", memory: memory)
        }
        let identifier = UUID()
        let report = ImageExecutionReport(
            requestID: identifier, profileID: "fixture", manifestSHA256: String(repeating: "a", count: 64),
            noiseSource: "fixture", diagnosticsEnabled: true)
        try await journal.record(
            ImageExecutionProgress(requestID: identifier, phase: .loading(.text)), report: report, memory: memory)
        try await journal.recordNative(stage: "function-load-before", asset: "text/text.aimodel", memory: memory)
        try await journal.record(
            ImageExecutionProgress(requestID: identifier, phase: .running(.denoiser, step: 3)), report: report,
            memory: memory)
        try await journal.recordNative(
            stage: "input-before", asset: "denoiser/denoiser.aimodel",
            tensor: ImageNativeTensorObservation(name: "value", shape: [1, 4], strides: [4, 1], minimumByteCount: 16),
            memory: memory)
        let saved = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: directory.appending(path: "native-checkpoint.json")))
                as? [String: Any])
        #expect(saved["requestID"] as? String == identifier.uuidString)
        #expect(saved["profileID"] as? String == "fixture")
        #expect(saved["manifestSHA256"] as? String == report.manifestSHA256)
        #expect(saved["sequence"] as? Int == 2)
        #expect(saved["checkpointSequence"] as? Int == 2)
        #expect(saved["component"] as? String == "denoiser")
        #expect(saved["step"] as? Int == 3)
        let tensor = try #require(saved["tensor"] as? [String: Any])
        #expect(tensor["shape"] as? [Int] == [1, 4])
        #expect(tensor["strides"] as? [Int] == [4, 1])
        #expect(tensor["minimumByteCount"] as? Int == 16)
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        return directory
    }

    private func checkpoint(_ directory: URL) throws -> [String: Any] {
        try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: directory.appending(path: "checkpoint.json")))
                as? [String: Any])
    }
}

private actor HeldImageJournal: ImageCheckpointSink {
    let events: AsyncStream<ImageExecutionPhase>
    private let sink: AsyncStream<ImageExecutionPhase>.Continuation
    private let journal: ImageExecutionJournal
    private let phase: ImageExecutionPhase?
    private let gate: AsyncStream<Void>?
    private let failing: ImageExecutionPhase?

    init(
        directory: URL, phase: ImageExecutionPhase? = nil, gate: AsyncStream<Void>? = nil,
        failing: ImageExecutionPhase? = nil
    ) {
        (events, sink) = AsyncStream.makeStream()
        journal = ImageExecutionJournal(directory: directory)
        self.phase = phase
        self.gate = gate
        self.failing = failing
    }

    func record(
        _ progress: ImageExecutionProgress, report: ImageExecutionReport, memory: ImageMemoryObservation
    ) async throws {
        if progress.phase == failing { throw ImageTestError.native }
        try await journal.record(progress, report: report, memory: memory)
        sink.yield(progress.phase)
        if progress.phase == phase, let gate {
            for await _ in gate { break }
        }
    }
}
