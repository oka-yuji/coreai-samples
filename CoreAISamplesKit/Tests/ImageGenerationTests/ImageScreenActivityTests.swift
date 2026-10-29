import Foundation
import ImageGenerationCore
import Testing

@testable import ImageGenerationRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ImageScreenActivityTests {
    @Test(
        "Overlapping owners restore the original idle timer only after the last valid release",
        arguments: [false, true])
    func ownership(original: Bool) {
        let screen = ImageScreenActivityState(disabled: original)
        let activity = screen.makeActivity()
        let first = activity.begin()
        let second = activity.begin()
        #expect(screen.disabled)
        activity.end(UUID())
        activity.end(first)
        activity.end(first)
        #expect(screen.disabled && screen.writes == [true])
        activity.end(second)
        activity.end(second)
        #expect(screen.disabled == original)
        #expect(screen.writes == [true, original])
    }

    @Test(
        "Accepted image requests hold the screen during execution and restore it on success or failure",
        arguments: [false, true])
    func requestLifetime(fails: Bool) async throws {
        let fixture = try ImageRuntimeFixture()
        defer { fixture.remove() }
        let backend = ImageFakeBackend()
        if fails { await backend.fail("call:text:0") }
        let (gate, resume) = AsyncStream<Void>.makeStream()
        defer { resume.finish() }
        await backend.hold("load:text", until: gate)
        var events = backend.events.makeAsyncIterator()
        let screen = ImageScreenActivityState()
        let service = LiveImageGenerationService(
            session: fixture.session(), pipeline: fixture.pipeline(backend: backend), activity: screen.makeActivity(),
            isPermitted: { true })
        let request = try fixture.request()
        let running = Task {
            defer { backend.finishEvents() }
            return try await service.generate(request, progress: { _ in })
        }
        try await imageEvent("load:text", from: &events)
        #expect(screen.disabled)
        resume.finish()
        if fails {
            await #expect(throws: ImageTestError.native) { try await running.value }
        } else {
            #expect(try await running.value.report.status == .completed)
        }
        #expect(!screen.disabled && screen.writes == [true, false])
        try await service.release()
    }
}

@MainActor
final class ImageScreenActivityState {
    var disabled: Bool
    var writes: [Bool] = []

    init(disabled: Bool = false) { self.disabled = disabled }

    func makeActivity() -> ImageScreenActivity {
        ImageScreenActivity(
            read: { self.disabled },
            write: {
                self.disabled = $0
                self.writes.append($0)
            })
    }
}
