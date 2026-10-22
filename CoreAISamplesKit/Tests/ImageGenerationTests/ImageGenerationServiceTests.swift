import Foundation
import ImageGenerationCore
import Testing

@testable import ImageGenerationRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ImageGenerationServiceTests {
    @Test("Repeated stop joins generation and permits an immediate new request without delivering the old result")
    func repeatedStopAndRestart() async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let backend = ImageFakeBackend()
        let (gate, resume) = AsyncStream<Void>.makeStream()
        defer { resume.finish() }
        await backend.hold("load:text", until: gate)
        var events = backend.events.makeAsyncIterator()
        let service = LiveImageGenerationService(
            session: fixture.session(), pipeline: fixture.pipeline(backend: backend), isPermitted: { true })
        let request = try fixture.request()
        let delivery = ImageProgressRecorder()
        let running = Task {
            defer { backend.finishEvents() }
            return try await service.generate(request, progress: delivery.receive)
        }
        try await imageEvent("load:text", from: &events)
        running.cancel()
        let firstStop = Task { try await service.stop() }
        let secondStop = Task { try await service.stop() }
        resume.finish()
        try await firstStop.value
        try await secondStop.value
        #expect(delivery.values.last?.phase == .loading(.text))
        let result = try await service.generate(fixture.request(), progress: delivery.receive)
        await #expect(throws: CancellationError.self) { try await running.value }
        #expect(result.report.status == .completed)
    }

    @Test("A revoked binding cannot publish progress or a successful result and a new owner can start")
    func bindingSwitch() async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let oldBackend = ImageFakeBackend()
        let permitted = ImagePermissionBox()
        let (gate, resume) = AsyncStream<Void>.makeStream()
        defer { resume.finish() }
        await oldBackend.hold("load:text", until: gate)
        var events = oldBackend.events.makeAsyncIterator()
        let old = LiveImageGenerationService(
            session: fixture.session(),
            pipeline: fixture.pipeline(backend: oldBackend, permission: { permitted.value }),
            isPermitted: { permitted.value })
        let delivery = ImageProgressRecorder()
        let request = try fixture.request()
        let running = Task {
            defer { oldBackend.finishEvents() }
            return try await old.generate(request, progress: delivery.receive)
        }
        try await imageEvent("load:text", from: &events)
        permitted.value = false
        resume.finish()
        await #expect(throws: ImageExecutionError.permissionDenied) { try await running.value }
        #expect(delivery.values.last?.phase == .loading(.text))
        #expect(await oldBackend.history == ["load:text", "unload:text"])
        try await old.release()
        let newBackend = ImageFakeBackend()
        let current = LiveImageGenerationService(
            session: fixture.session(number: 2), pipeline: fixture.pipeline(backend: newBackend), isPermitted: { true })
        #expect(try await current.generate(fixture.request(), progress: { _ in }).report.status == .completed)
        await #expect(throws: ImageExecutionError.released) { try await old.generate(request, progress: { _ in }) }
    }
}

@MainActor
private final class ImagePermissionBox {
    var value = true
}

@MainActor
private final class ImageProgressRecorder {
    var values: [ImageExecutionProgress] = []
    func receive(_ value: ImageExecutionProgress) { values.append(value) }
}
