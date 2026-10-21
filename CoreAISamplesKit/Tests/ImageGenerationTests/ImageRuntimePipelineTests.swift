import Foundation
import ImageGenerationCore
import Testing

@testable import ImageGenerationRuntime

@Suite(.timeLimit(.minutes(1)))
struct ImageRuntimePipelineTests {
    @Test("A request withdrawn while waiting for admission cannot validate or load the model")
    func withdrawnAdmission() async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let backend = ImageFakeBackend()
        let pipeline = fixture.pipeline(backend: backend)
        let request = try fixture.request()
        await #expect(throws: CancellationError.self) {
            try await pipeline.generate(
                request, isCurrent: { false },
                progress: { _ in
                    Issue.record("Withdrawn requests must not deliver progress")
                })
        }
        #expect(await backend.history.isEmpty)
        #expect(await pipeline.report?.status == .cancelled)
    }

    @Test(
        "Load and native call failures release partial resources while nonfinite outputs are traced before rejection",
        arguments: ["load:text", "call:text:0", "nonfinite"])
    func failedOperations(_ failure: String) async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let backend = ImageFakeBackend()
        if failure == "nonfinite" { await backend.returnNonfinite() } else { await backend.fail(failure) }
        let pipeline = fixture.pipeline(backend: backend)
        let trace = ImageFakeTrace()
        let request = try fixture.request()
        if failure == "nonfinite" {
            await #expect(throws: ImageExecutionError.invalidTensor("hidden")) {
                try await pipeline.generate(request, trace: trace, progress: { _ in })
            }
            #expect(await trace.names.contains("text-outputs"))
        } else {
            await #expect(throws: ImageTestError.native) {
                try await pipeline.generate(request, trace: trace, progress: { _ in })
            }
        }
        #expect(await backend.history.last == "unload:text")
        #expect(await pipeline.report?.status == .failed)
    }

    @Test(
        "Generation and editing release each component before loading the next and execute exactly eight steps",
        arguments: [false, true])
    func orderedExecution(_ editing: Bool) async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let backend = ImageFakeBackend()
        let pipeline = fixture.pipeline(backend: backend)
        let trace = ImageFakeTrace()
        let request = try fixture.request(editing: editing)
        let result = try await pipeline.generate(request, trace: trace, progress: { _ in })
        let kinds =
            editing
            ? ["vision", "text", "vae_encoder", "denoiser", "vae_decoder"] : ["text", "denoiser", "vae_decoder"]
        let expected = kinds.flatMap { kind in
            ["load:" + kind] + (0..<(kind == "denoiser" ? 8 : 1)).map { "call:\(kind):\($0)" } + ["unload:" + kind]
        }
        #expect(await backend.history == expected)
        #expect(result.requestID == request.id)
        #expect(result.pixels.width == 512 && result.pixels.rgba.allSatisfy { $0 == 128 })
        #expect(result.report.status == .completed)
        #expect(result.report.diagnosticsEnabled)
        #expect(result.report.phases.filter { $0.component == .denoiser && $0.operation == .inference }.count == 8)
        #expect(await trace.names.contains("initial-latents"))
        #expect(await trace.names.contains("latent-7"))
        #expect(await trace.names.contains("vae_decoder-outputs"))
    }

    @Test("Stopping a load joins it before release and never starts inference")
    func stopDuringLoad() async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let backend = ImageFakeBackend()
        let (gate, resume) = AsyncStream<Void>.makeStream()
        defer { resume.finish() }
        await backend.hold("load:text", until: gate)
        var events = backend.events.makeAsyncIterator()
        let pipeline = fixture.pipeline(backend: backend)
        let request = try fixture.request()
        let running = Task {
            defer { backend.finishEvents() }
            return try await pipeline.generate(request, progress: { _ in })
        }
        try await imageEvent("load:text", from: &events)
        running.cancel()
        let stop = Task { try await pipeline.stop() }
        #expect(await backend.history == ["load:text"])
        await #expect(throws: ImageExecutionError.busy) { try await pipeline.generate(request, progress: { _ in }) }
        resume.finish()
        await #expect(throws: CancellationError.self) { try await running.value }
        try await stop.value
        #expect(await backend.history == ["load:text", "unload:text"])
        #expect(try await pipeline.generate(fixture.request(), progress: { _ in }).report.status == .completed)
    }

    @Test(
        "Task cancellation during native execution joins the call before release and permits another request",
        arguments: ["call:text:0", "call:denoiser:3", "call:vae_decoder:0"])
    func cancellationDuringCall(_ event: String) async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let backend = ImageFakeBackend()
        let (gate, resume) = AsyncStream<Void>.makeStream()
        defer { resume.finish() }
        await backend.hold(event, until: gate)
        var events = backend.events.makeAsyncIterator()
        let pipeline = fixture.pipeline(backend: backend)
        let request = try fixture.request()
        let running = Task {
            defer { backend.finishEvents() }
            return try await pipeline.generate(request, progress: { _ in })
        }
        try await imageEvent(event, from: &events)
        running.cancel()
        #expect(await backend.history.last == event)
        resume.finish()
        await #expect(throws: CancellationError.self) { try await running.value }
        #expect(await backend.history.last == "unload:" + event.split(separator: ":")[1])
        let result = try await pipeline.generate(fixture.request(), progress: { _ in })
        #expect(result.report.status == .completed)
    }

    @Test("Unload failure retains the resource and blocks another generation until explicit cleanup succeeds")
    func releaseFailureBlocksReuse() async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let backend = ImageFakeBackend()
        await backend.fail("call:text:0")
        await backend.fail("unload:text")
        let pipeline = fixture.pipeline(backend: backend)
        let request = try fixture.request()
        await #expect(throws: ImageExecutionError.self) { try await pipeline.generate(request, progress: { _ in }) }
        #expect(await pipeline.report?.status == .releaseUnconfirmed)
        let history = await backend.history
        await #expect(throws: ImageExecutionError.self) { try await pipeline.generate(request, progress: { _ in }) }
        #expect(await backend.history == history)
        await #expect(throws: ImageExecutionError.self) { try await pipeline.stop() }
        await backend.clearFailures()
        try await pipeline.stop()
        #expect(try await pipeline.generate(request, progress: { _ in }).report.status == .completed)
    }

    @Test("A fresh memory check refuses inference after a successful load and releases that load")
    func budgetChangesBeforeInference() async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let backend = ImageFakeBackend()
        let memory = ImageTestMemory()
        let pipeline = fixture.pipeline(backend: backend, memory: memory)
        let request = try fixture.request()
        await #expect(throws: ImageExecutionError.self) {
            try await pipeline.generate(request) { value in
                if value.phase == .running(.text, step: nil) { memory.setAvailable(1) }
            }
        }
        #expect(await backend.history == ["load:text", "unload:text"])
        #expect(await pipeline.report?.status == .failed)
    }

    @Test("Revoked permission and invalid inputs are rejected before any model load")
    func preflightRefusal() async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let backend = ImageFakeBackend()
        let pipeline = fixture.pipeline(backend: backend, permission: { false })
        let request = try fixture.request()
        await #expect(throws: ImageExecutionError.permissionDenied) {
            try await pipeline.generate(request, progress: { _ in })
        }
        let permitted = fixture.pipeline(backend: backend)
        let invalid = ImageGenerationRequest(id: UUID(), prompt: " ", seed: 1)
        await #expect(throws: ImageGenerationError.invalidTokens) {
            try await permitted.generate(invalid, progress: { _ in })
        }
        #expect(await backend.history.isEmpty)
    }

    @Test("Permanent release prevents the old binding from starting again")
    func permanentRelease() async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let backend = ImageFakeBackend()
        let pipeline = fixture.pipeline(backend: backend)
        try await pipeline.stop(releasing: true)
        try await pipeline.stop(releasing: true)
        await #expect(throws: ImageExecutionError.released) {
            try await pipeline.generate(fixture.request(), progress: { _ in })
        }
        #expect(await backend.history.isEmpty)
    }
}
