import Foundation
import OCRCore
import OCRFeature
import Testing

@testable import OCRDemoRuntime

@MainActor
struct OCRRunnerTests {
    @Test("The runner returns the selected image, loaded stage and matching page document")
    func success() async throws {
        let fixture = try pageFixture()
        let service = FakeOCRService()
        service.fixture = fixture
        let runner = OCRRunner(service: service)
        var actions: [OCRAction] = []
        await runner.run(.select(1, URL(filePath: "/fixture/page.png"))) { actions.append($0) }
        await runner.run(.load(2, URL(filePath: "/fixture/model"))) { actions.append($0) }
        await runner.run(.read(2, fixture.input)) { actions.append($0) }
        #expect(actions == [.selected(1, fixture.input), .loaded(2), .completed(2, fixture.document)])
        #expect(service.loads == 1)
    }

    @Test("The runner reports load failures and cancellation without swallowing a terminal state")
    func failures() async {
        let service = FakeOCRService()
        let runner = OCRRunner(service: service)
        var actions: [OCRAction] = []
        service.cancel = false
        await runner.run(.load(1, URL(filePath: "/fixture/model"))) { actions.append($0) }
        #expect(actions == [.failed(1, "Fixture failure")])
        service.cancel = true
        actions.removeAll()
        await runner.run(.load(2, URL(filePath: "/fixture/model"))) { actions.append($0) }
        #expect(actions == [.cancelled(2)])
        actions.removeAll()
        await runner.run(.release(3)) { actions.append($0) }
        #expect(actions == [.stopped(3)] && service.releases == 1)
    }

    @Test("A revoked owner cannot reach the service and can still release its previous resources")
    func permission() async {
        let service = FakeOCRService()
        let runner = OCRRunner(service: service, isPermitted: { false })
        var actions: [OCRAction] = []
        await runner.run(.load(1, URL(filePath: "/fixture/model"))) { actions.append($0) }
        #expect(service.loads == 0)
        #expect(actions == [.failed(1, "The selected model is not available right now. Try again.")])
        await runner.release()
        #expect(service.releases == 1)
    }
}

@MainActor
private final class FakeOCRService: OCRService {
    var fixture: (input: OCRInput, document: OCRDocument)?
    var cancel = false
    var loads = 0
    var releases = 0

    func image(at url: URL) async throws -> OCRInput {
        guard let fixture else { throw Failure() }
        return fixture.input
    }
    func load(bundle: URL) async throws {
        loads += 1
        if cancel { throw CancellationError() }
        if fixture != nil { return }
        throw Failure()
    }
    func read(_ input: OCRInput) async throws -> OCRDocument {
        guard let fixture else { throw Failure() }
        return fixture.document
    }
    func release() async { releases += 1 }

    private struct Failure: LocalizedError {
        var errorDescription: String? { "Fixture failure" }
    }
}
