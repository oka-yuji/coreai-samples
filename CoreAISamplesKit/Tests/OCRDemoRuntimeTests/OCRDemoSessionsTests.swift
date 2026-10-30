import FeatureStore
import Foundation
import ModelLibraryFeature
import OCRFeature
import Testing

@testable import OCRDemoRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct OCRDemoSessionsTests {
    @Test("OCR owner handoff retains only the selected image and retires the previous execution permission")
    func handoff() async throws {
        let fixture = try pageFixture()
        let events = AsyncStream<(LibraryTicket, OCRAction)>.makeStream()
        var services: [SessionService] = []
        let sessions = OCRDemoSessions(
            makeService: {
                let service = SessionService(input: fixture.input, document: fixture.document)
                services.append(service)
                return service
            }, observe: { events.continuation.yield(($0, $1)) })
        let first = session(owner: UUID())
        let second = session(owner: UUID())
        let firstBinding = sessions.bind(first, permits: { true })
        let old = try #require(firstBinding)
        #expect(sessions.bind(first, permits: { true }) === old)
        #expect(sessions.bind(second, permits: { true }) == nil && services.count == 1)
        old.send(.selectImage(URL(filePath: "/fixture/page.png")))
        var observed = events.stream.makeAsyncIterator()
        while let event = await observed.next() {
            if event.0 == first.id, case .selected = event.1 { break }
        }
        await sessions.stop(first)
        while let event = await observed.next() {
            if event.0 == first.id, case .stopped(2) = event.1 { break }
        }
        let secondBinding = sessions.bind(second, permits: { true })
        let current = try #require(secondBinding)
        #expect(current !== old && current.state.input == fixture.input && current.state.document == nil)
        old.send(.appeared)
        old.send(.read)
        while let event = await observed.next() {
            if event.0 == first.id, case .failed = event.1 { break }
        }
        #expect(services[0].loads == 0)
        current.send(.read)
        while let event = await observed.next() {
            if event.0 == second.id, case .completed = event.1 { break }
        }
        #expect(current.state.document == fixture.document && services[1].loads == 1)
        await sessions.stop(first)
        #expect(current.state.presented)
        old.send(.ended)
        while let event = await observed.next() {
            if event.0 == first.id, case .stopped(4) = event.1 { break }
        }
        await sessions.stop(second)
        while let event = await observed.next() {
            if event.0 == second.id, case .stopped(2) = event.1 { break }
        }
        events.continuation.finish()
    }

    @Test("A new OCR binding remains blocked until the previous service acknowledges release")
    func delayedStop() async throws {
        let fixture = try pageFixture()
        let service = SessionService(input: fixture.input, document: fixture.document)
        service.delay = true
        let events = AsyncStream<(LibraryTicket, OCRAction)>.makeStream()
        var created = false
        let sessions = OCRDemoSessions(
            makeService: {
                if !created {
                    created = true
                    return service
                }
                return SessionService(input: fixture.input, document: fixture.document)
            }, observe: { events.continuation.yield(($0, $1)) })
        let first = session(owner: UUID())
        let next = session(owner: UUID())
        let firstBinding = sessions.bind(first, permits: { true })
        let store = try #require(firstBinding)
        #expect(sessions.bind(next, permits: { false }) == nil)
        let stopping = Task { await sessions.stop(first) }
        var started = service.releaseStarted.makeAsyncIterator()
        _ = await started.next()
        #expect(store.state.phase == .stopping)
        #expect(sessions.bind(next, permits: { true }) == nil)
        service.finishRelease()
        await stopping.value
        var observed = events.stream.makeAsyncIterator()
        while let event = await observed.next() {
            if event.0 == first.id, case .stopped(1) = event.1 { break }
        }
        weak var unpresented = sessions.bind(next, permits: { true })
        #expect(unpresented != nil)
        await sessions.stop(next)
        #expect(unpresented == nil)
        events.continuation.finish()
    }

    private func session(owner: UUID) -> ModelDemoSession {
        ModelDemoSession(
            id: LibraryTicket(owner: owner, number: 1), selection: TeleOCRDemoCatalog.selection,
            resource: ValidatedModelResource(
                modelID: "teleocr", artifactID: "teleocr-fp16", contract: "teleocr-dynamic-fp16-v1",
                revision: String(repeating: "a", count: 64), upstreamRevision: nil,
                directory: URL(filePath: "/fixture/model"), sizeBytes: 0, sourceKind: .localManifest))
    }
}

@MainActor
private final class SessionService: OCRService {
    let input: OCRInput
    let document: OCRDocument
    let releaseStarted: AsyncStream<Void>
    private let releaseContinuation: AsyncStream<Void>.Continuation
    private var waiting: [CheckedContinuation<Void, Never>] = []
    var loads = 0
    var delay = false

    init(input: OCRInput, document: OCRDocument) {
        self.input = input
        self.document = document
        (releaseStarted, releaseContinuation) = AsyncStream<Void>.makeStream()
    }
    func image(at url: URL) async throws -> OCRInput { input }
    func load(bundle: URL) async throws { loads += 1 }
    func read(_ input: OCRInput) async throws -> OCRDocument { document }
    func release() async {
        if delay {
            releaseContinuation.yield(())
            await withCheckedContinuation { waiting.append($0) }
        }
    }
    func finishRelease() {
        delay = false
        let pending = waiting
        waiting.removeAll()
        for item in pending { item.resume() }
        releaseContinuation.finish()
    }
}
