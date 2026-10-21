import Dispatch
import FeatureStore
import Foundation
import ImageDiagnosticFeature
import ImageGenerationCore
import Testing

@testable import ImageGenerationRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ImageDiagnosticRunnerTests {
    @Test("A preparation error remains a failure when cancellation and stop arrive before the file actor resumes")
    func cancelledPreparationFailure() async throws {
        let root = try diagnosticTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let specification = try diagnosticSpecification(root)
        let output = root.appending(path: "image-check-fixture")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        let sentinel = output.appending(path: "existing.txt")
        try Data("keep".utf8).write(to: sentinel)
        let files = ImageDiagnosticFiles()
        let blocker = DispatchSemaphore(value: 0)
        defer { blocker.signal() }
        let entered = AsyncStream<Void>.makeStream()
        let holder = Task { await files.holdDiagnosticFixture(blocker, entered: entered.continuation) }
        var iterator = entered.stream.makeAsyncIterator()
        _ = await iterator.next(isolation: #isolation)
        let environmentRead = DiagnosticGate()
        let stopStarted = DiagnosticGate()
        let terminal = DiagnosticGate()
        var factories = 0
        let runner = ImageDiagnosticRunner(
            files: files,
            environment: {
                environmentRead.open()
                return ("fixture-device", "fixture-os")
            },
            makeService: { _, _, _ in
                factories += 1
                return DiagnosticFakeService()
            })
        let store = Store<ImageDiagnosticFeature>(initialState: ImageDiagnosticState()) { effect, send in
            if case .stop = effect { stopStarted.open() }
            await runner.run(effect) { action in
                send(action)
                if case .stopped = action { terminal.open() }
            }
        }
        store.send(.start(specification))
        await environmentRead.wait()
        store.send(.stop)
        await stopStarted.wait()
        #expect(store.state.phase == .stopping)
        blocker.signal()
        await holder.value
        await terminal.wait()
        #expect(store.state.phase == .finished)
        #expect(store.state.outcome?.status == .failed)
        #expect(store.state.message.contains("17"))
        #expect(factories == 0)
        #expect(try Data(contentsOf: sentinel) == Data("keep".utf8))
    }

    @Test("Diagnostic stop joins the native operation cleanup and final record before acknowledging repeated stop")
    func joinedStop() async throws {
        let root = try diagnosticTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let specification = try diagnosticSpecification(root)
        let service = DiagnosticFakeService()
        let terminal = DiagnosticGate()
        let runner = ImageDiagnosticRunner(
            files: ImageDiagnosticFiles(), environment: { ("fixture-device", "fixture-os") },
            makeService: { _, _, _ in service })
        let store = Store<ImageDiagnosticFeature>(initialState: ImageDiagnosticState()) { effect, send in
            await runner.run(effect) { action in
                send(action)
                if case .stopped = action { terminal.open() }
            }
        }
        store.send(.start(specification))
        await service.entered.wait()
        store.send(.stop)
        await service.stopEntered.wait()
        store.send(.stop)
        #expect(store.state.phase == .stopping)
        #expect(service.stopCalls == 1)
        #expect(
            !FileManager.default.fileExists(atPath: root.appending(path: "image-check-fixture/completion.json").path))
        service.cleanup.open()
        await terminal.wait()
        #expect(store.state.phase == .finished)
        #expect(store.state.outcome?.status == .cancelled)
        #expect(service.releaseCalls == 1)
        let completion = try #require(
            JSONSerialization.jsonObject(
                with: Data(contentsOf: root.appending(path: "image-check-fixture/completion.json"))) as? [String: Any])
        #expect(completion["status"] as? String == "cancelled")
        #expect(!FileManager.default.fileExists(atPath: root.appending(path: "image-check-fixture/output.png").path))
    }

    @Test(
        "Native and recording failures survive cleanup retry without converting the run to success",
        arguments: [false, true])
    func recordingFailure(removeDirectory: Bool) async throws {
        let root = try diagnosticTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let specification = try diagnosticSpecification(root)
        let service = DiagnosticFakeService()
        if removeDirectory { service.failureDirectory = root.appending(path: "image-check-fixture") }
        service.failGeneration = true
        service.cleanupFailure = true
        service.finish.open()
        service.cleanup.open()
        let terminal = DiagnosticGate()
        let recovered = DiagnosticGate()
        let runner = ImageDiagnosticRunner(
            files: ImageDiagnosticFiles(), environment: { ("fixture-device", "fixture-os") },
            makeService: { _, _, _ in service })
        let store = Store<ImageDiagnosticFeature>(initialState: ImageDiagnosticState()) { effect, send in
            await runner.run(effect) { action in
                send(action)
                if case .completed = action { terminal.open() }
                if case .failed = action { terminal.open() }
                if case .stopped = action { recovered.open() }
            }
        }
        store.send(.start(specification))
        await terminal.wait()
        #expect(store.state.phase == .cleanupFailed)
        #expect(store.state.outcome?.status == .releaseUnconfirmed)
        #expect(store.state.message.contains("invalidResource"))
        #expect(store.state.message.contains("fixture cleanup"))
        if removeDirectory { #expect(store.state.message.contains("recording")) }
        service.cleanupFailure = false
        store.send(.stop)
        await recovered.wait()
        #expect(store.state.phase == .finished)
        #expect(store.state.outcome?.status == .failed)
        #expect(store.state.message.contains("Cleanup completed after"))
        #expect(service.releaseCalls == 2)
        let recoveryFile = root.appending(path: "image-check-fixture/cleanup-retry-1.json")
        #expect(FileManager.default.fileExists(atPath: recoveryFile.path) == !removeDirectory)
    }
}

extension ImageDiagnosticFiles {
    fileprivate func holdDiagnosticFixture(_ blocker: DispatchSemaphore, entered: AsyncStream<Void>.Continuation) {
        entered.yield(())
        blocker.wait()
    }
}

@MainActor
private final class DiagnosticGate {
    private var opened = false
    private var waiting: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        if opened { return }
        await withCheckedContinuation { waiting.append($0) }
    }
    func open() {
        opened = true
        let pending = waiting
        waiting.removeAll()
        for waiter in pending { waiter.resume() }
    }
}

@MainActor
private final class DiagnosticFakeService: ImageGenerationService {
    let entered = DiagnosticGate()
    let stopEntered = DiagnosticGate()
    let finish = DiagnosticGate()
    let cleanup = DiagnosticGate()
    var stopCalls = 0
    var releaseCalls = 0
    var failureDirectory: URL?
    var cleanupFailure = false
    var failGeneration = false
    private var report: ImageExecutionReport?

    func generate(
        _ request: ImageGenerationRequest, trace: (any ImageTraceSink)?,
        progress: @MainActor @Sendable (ImageExecutionProgress) -> Void
    ) async throws -> ImageGenerationResult {
        report = ImageExecutionReport(
            requestID: request.id, profileID: "fixture-profile",
            manifestSHA256: String(repeating: "a", count: 64), noiseSource: "fixture", diagnosticsEnabled: false)
        progress(ImageExecutionProgress(requestID: request.id, phase: .loading(.text)))
        entered.open()
        await finish.wait()
        do {
            try Task.checkCancellation()
            if let failureDirectory {
                try FileManager.default.removeItem(at: failureDirectory)
                throw ImageExecutionError.invalidResource
            }
            if failGeneration { throw ImageExecutionError.invalidResource }
            let pixels = try ImageGenerationPixels(
                width: 512, height: 512, rgba: Array(repeating: 255, count: 512 * 512 * 4))
            report?.status = .completed
            return ImageGenerationResult(requestID: request.id, pixels: pixels, report: try #require(report))
        } catch {
            report?.status = error is CancellationError ? .cancelled : .failed
            report?.error = String(describing: error)
            throw error
        }
    }
    func stop() async throws {
        stopCalls += 1
        stopEntered.open()
        finish.open()
        await cleanup.wait()
    }
    func release() async throws {
        releaseCalls += 1
        await cleanup.wait()
        if cleanupFailure { throw ImageExecutionError.releaseUnconfirmed("fixture cleanup") }
    }
    func latestReport() async -> ImageExecutionReport? { report }
}
