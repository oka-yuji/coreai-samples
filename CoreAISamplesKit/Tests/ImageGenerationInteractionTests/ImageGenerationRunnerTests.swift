import Foundation
import ImageGenerationCore
import ImageGenerationFeature
import ImageGenerationRuntime
import Testing

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ImageGenerationRunnerTests {
    @Test("A permitted generation delivers progress and a complete PNG while denial performs no native operation")
    func generationAndDenial() async throws {
        let service = ImageInteractionService()
        let runner = ImageGenerationRunner(service: service, isPermitted: { true })
        let request = ImageGenerationRequest(id: UUID(), prompt: "image", seed: 4)
        let ticket = ImageGenerationTicket(owner: UUID(), sequence: 1)
        var actions: [ImageGenerationAction] = []
        await runner.run(.generate(ticket, request)) { actions.append($0) }
        #expect(service.requests.count == 1)
        #expect(actions.count == 3)
        guard case .generated(let received, let asset)? = actions.last else {
            Issue.record("A completed image must be delivered")
            return
        }
        #expect(received == ticket && asset.id == request.id && asset.seed == 4 && !asset.png.isEmpty)
        let denied = ImageGenerationRunner(service: service, isPermitted: { false })
        actions.removeAll()
        await denied.run(.generate(ticket, request)) { actions.append($0) }
        #expect(service.requests.count == 1)
        guard case .failed(let rejected, .unavailable)? = actions.last else {
            Issue.record("Denied permission must be reported")
            return
        }
        #expect(rejected == ticket)
    }

    @Test("Native failure and unconfirmed stop reach the matching failure actions")
    func failures() async {
        let service = ImageInteractionService()
        service.failure = true
        service.stopFailure = true
        let runner = ImageGenerationRunner(service: service, isPermitted: { true })
        let ticket = ImageGenerationTicket(owner: UUID(), sequence: 1)
        var failure: ImageGenerationFailure?
        await runner.run(.generate(ticket, ImageGenerationRequest(id: UUID(), prompt: "image", seed: 1))) {
            if case .failed(let value, let error) = $0, value == ticket { failure = error }
        }
        #expect(failure == .invalidModel)
        var stopped = false
        await runner.run(.stop(ticket)) { if case .stopFailed(let value) = $0, value == ticket { stopped = true } }
        #expect(stopped)
    }

    @Test("A canceled generation ignores delayed progress and results and stop waits for its completion")
    func canceledGeneration() async throws {
        let service = ImageInteractionService()
        let (gate, resume) = AsyncStream<Void>.makeStream()
        defer { resume.finish() }
        service.gate = gate
        var events = service.events.makeAsyncIterator()
        let runner = ImageGenerationRunner(service: service, isPermitted: { true })
        let ticket = ImageGenerationTicket(owner: UUID(), sequence: 1)
        let recorder = ImageInteractionRecorder()
        let task = Task {
            await runner.run(
                .generate(ticket, ImageGenerationRequest(id: UUID(), prompt: "image", seed: 1)), send: recorder.receive)
        }
        try await imageInteractionEvent("generate", from: &events)
        task.cancel()
        let stop = Task { await runner.run(.stop(ticket), send: recorder.receive) }
        try await imageInteractionEvent("stop", from: &events)
        #expect(recorder.actions.count == 1)
        resume.finish()
        await task.value
        await stop.value
        #expect(recorder.actions.count == 2)
        guard case .stopped(let value)? = recorder.actions.last else {
            Issue.record("Stop must be acknowledged after the old call finishes")
            return
        }
        #expect(value == ticket)
    }
}

@MainActor
private final class ImageInteractionRecorder {
    var actions: [ImageGenerationAction] = []
    func receive(_ action: ImageGenerationAction) { actions.append(action) }
}
