import FeatureStore
import Foundation
import ObjectFinderCore
import ObjectFinderFeature
import Testing

@testable import ObjectFinderRuntime

@MainActor
private final class ExecutionPermission {
    var granted = true
    var checks = 0

    func check() -> Bool {
        checks += 1
        return granted
    }
}

@MainActor
private struct ManagedFinder {
    let store: Store<ObjectFinderFeature>
    let runner: ObjectFinderRunner
    let camera = FinderDevicesFake()
    let encoder: ReusedEncoder
    let permission = ExecutionPermission()
    let actions: AsyncStream<ObjectFinderAction>
    let requests: AsyncStream<Void>

    init(folder: URL, retaining previous: ObjectFinderState? = nil) throws {
        let encoder = try ReusedEncoder()
        let finder = ObjectRegionFinder(
            embedding: .init(
                load: { await encoder.load($0) }, text: { await encoder.text($0) },
                image: { _ in encoder.keys }, close: { await encoder.close() }), propose: { _ in [] })
        let requests = AsyncStream<Void>.makeStream()
        var devices = camera.devices
        let closeCamera = devices.closeFinder
        devices.prepare = { try await finder.prepare(folder: $0, query: $1) }
        devices.closeFinder = {
            try await finder.close()
            try await closeCamera()
        }
        devices.requestFrame = { requests.continuation.yield(()) }
        let runner = ObjectFinderRunner(service: LiveObjectFinderService(devices: devices), permits: permission.check)
        let actions = AsyncStream<ObjectFinderAction>.makeStream()
        store = Store(initialState: .managed(folder: folder, retaining: previous)) { effect, send in
            await runner.run(effect) { action in
                send(action)
                actions.continuation.yield(action)
            }
        }
        self.runner = runner
        self.encoder = encoder
        self.actions = actions.stream
        self.requests = requests.stream
    }

    func release() async throws {
        try await runner.release(store.state.session, send: store.send)
    }
}

private func next(
    _ actions: inout AsyncStream<ObjectFinderAction>.Iterator, where matches: (ObjectFinderAction) -> Bool
) async {
    while let action = await actions.next(isolation: #isolation) {
        if matches(action) { return }
    }
    Issue.record("The expected action was not sent")
}

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ObjectFinderManagedTests {
    enum LibraryExit: CaseIterable {
        case back, background, interruption
    }

    private let selected = URL(filePath: "/models/embeddinggemma2-selected")
    private let replacement = URL(filePath: "/models/embeddinggemma2-replacement")
    private let candidate = ObjectCandidate(
        id: 0, region: .init(x: 0.2, y: 0.2, width: 0.3, height: 0.3), similarity: 0.8)

    @Test(
        "CAMERA-01 Only an explicit Start for the selected artifact on a valid session requests permission, starts capture and runs inference"
    )
    func explicitStart() async throws {
        let finder = try ManagedFinder(folder: selected)
        var actions = finder.actions.makeAsyncIterator()
        var requests = finder.requests.makeAsyncIterator()
        for action in [
            ObjectFinderAction.appeared, .poseChanged(.landscapeLeft), .queryChanged("glasses"), .thresholdChanged(0.7),
            .folderSelected(replacement), .stop, .disappeared, .appeared,
        ] {
            finder.store.send(action)
        }
        #expect(finder.store.state.folder == selected && finder.store.state.execution == .managed)
        #expect(finder.permission.checks == 0 && finder.camera.authorizations == 0 && finder.camera.angles.isEmpty)
        #expect(await finder.encoder.loads.isEmpty)

        finder.permission.granted = false
        finder.store.send(.start)
        await next(&actions) { if case .failed(1, .notPermitted) = $0 { true } else { false } }
        #expect(finder.store.state.phase == .idle && finder.store.state.failure == .notPermitted)
        #expect(finder.camera.authorizations == 0 && finder.camera.angles.isEmpty)
        #expect(await finder.encoder.loads.isEmpty)

        finder.permission.granted = true
        finder.store.send(.start)
        _ = await requests.next()
        #expect(finder.store.state.phase == .searching && finder.store.state.failure == nil)
        #expect(finder.permission.checks == 2 && finder.camera.authorizations == 1 && finder.camera.angles == [0])
        #expect(await finder.encoder.loads == [selected])
        #expect(await finder.encoder.queries == ["glasses"])
        try await finder.release()
    }

    @Test(
        "CAMERA-02 A query change in the same session reuses the prepared model, while leaving Models fully stops and releases the camera, inference and model",
        arguments: [false, true])
    func reuseOrRelease(gateClosesBeforeRestart: Bool) async throws {
        let finder = try ManagedFinder(folder: selected)
        var actions = finder.actions.makeAsyncIterator()
        var requests = finder.requests.makeAsyncIterator()
        finder.store.send(.queryChanged("glasses"))
        finder.store.send(.start)
        _ = await requests.next()
        finder.permission.granted = !gateClosesBeforeRestart
        finder.store.send(.queryChanged("keys"))
        finder.store.send(.applyQuery)
        if gateClosesBeforeRestart {
            await next(&actions) { if case .failed(2, .notPermitted) = $0 { true } else { false } }
            #expect(finder.store.state.phase == .idle && finder.camera.angles == [90])
            #expect(await finder.encoder.queries == ["glasses"])
        } else {
            _ = await requests.next()
            #expect(finder.store.state.phase == .searching && finder.store.state.submittedQuery == "keys")
            #expect(finder.camera.angles == [90, 90])
            #expect(await finder.encoder.queries == ["glasses", "keys"])
        }
        #expect(finder.store.state.session == 2 && finder.camera.closes == 0)
        #expect(await finder.encoder.loads == [selected])
        #expect(await finder.encoder.closeCancellation.isEmpty)

        try await finder.release()
        #expect(finder.store.state.phase == .idle && finder.store.state.frame == nil)
        #expect(!finder.store.state.releasePending && finder.camera.frames == nil && finder.camera.closes == 1)
        #expect(await finder.encoder.loads == [selected])
        #expect(await finder.encoder.closeCancellation == [false])
        try await finder.release()
        #expect(finder.camera.closes == 1 && finder.store.state.phase == .idle)
    }

    @Test(
        "CAMERA-03 A failed release keeps Start rejected, and only an acknowledged retry allows another search")
    func failedReleaseReducer() {
        var state = ObjectFinderState.managed(folder: selected)
        state.query = "glasses"
        _ = ObjectFinderFeature.reduce(&state, .start)
        _ = ObjectFinderFeature.reduce(&state, .started(1))
        #expect(
            ObjectFinderFeature.reduce(&state, .stop) == [
                .cancelAll, .run(.stop(1, retainingModel: false), id: .control),
            ])
        #expect(ObjectFinderFeature.reduce(&state, .start).isEmpty && state.phase == .stopping)
        _ = ObjectFinderFeature.reduce(&state, .stopFailed(1, .operation("release failed")))
        #expect(state.phase == .idle && state.releasePending && !state.canStart && state.canStop)
        #expect(ObjectFinderFeature.reduce(&state, .start).isEmpty && state.session == 1)
        #expect(ObjectFinderFeature.reduce(&state, .stopped(0)).isEmpty && state.releasePending)
        #expect(
            ObjectFinderFeature.reduce(&state, .disappeared) == [
                .cancelAll, .run(.stop(1, retainingModel: false), id: .control),
            ])
        #expect(state.phase == .stopping && state.failure == nil && !state.canStop)
        _ = ObjectFinderFeature.reduce(&state, .stopFailed(1, .operation("release failed")))
        _ = ObjectFinderFeature.reduce(&state, .appeared)
        _ = ObjectFinderFeature.reduce(&state, .stop)
        _ = ObjectFinderFeature.reduce(&state, .stopped(1))
        #expect(state.phase == .idle && !state.releasePending && state.canStart)
        #expect(
            ObjectFinderFeature.reduce(&state, .start) == [
                .run(
                    .search(2, selected, "glasses", 0.67, .standard, .init(session: 2, revision: 0, angle: 90)),
                    id: .search)
            ])
        _ = ObjectFinderFeature.reduce(&state, .failed(2, .cameraInterrupted))
        _ = ObjectFinderFeature.reduce(&state, .stopFailed(2, .operation("release failed")))
        #expect(state.releasePending && state.failure == .operation("release failed") && !state.canStart)
        _ = ObjectFinderFeature.reduce(&state, .stopFailed(1, .cameraDenied))
        #expect(state.failure == .operation("release failed"))

        var standalone = ObjectFinderState(folder: selected)
        standalone.query = "glasses"
        for action in [ObjectFinderAction.start, .started(1), .stop, .stopFailed(1, .operation("release failed"))] {
            _ = ObjectFinderFeature.reduce(&standalone, action)
        }
        _ = ObjectFinderFeature.reduce(&standalone, .folderSelected(replacement))
        #expect(standalone.folder == selected && standalone.failure == .operation("release failed"))
        _ = ObjectFinderFeature.reduce(&standalone, .stop)
        _ = ObjectFinderFeature.reduce(&standalone, .stopped(1))
        _ = ObjectFinderFeature.reduce(&standalone, .folderSelected(replacement))
        #expect(standalone.folder == replacement && !standalone.releasePending)
    }

    @Test(
        "CAMERA-03 Back navigation, background, interruption, stop failure and retry keep resource ownership and reject restarts, old frames and notifications before acknowledgement",
        arguments: LibraryExit.allCases)
    func ownershipAcrossExits(_ exit: LibraryExit) async throws {
        let finder = try ManagedFinder(folder: selected)
        var actions = finder.actions.makeAsyncIterator()
        var requests = finder.requests.makeAsyncIterator()
        finder.store.send(.queryChanged("glasses"))
        finder.store.send(.start)
        _ = await requests.next()
        let frame = ObjectFinderFrame(candidates: [candidate], seconds: 0.1)
        finder.camera.releaseError = .operation("release failed")
        switch exit {
        case .back: finder.store.send(.disappeared)
        case .background: finder.store.send(.backgrounded)
        case .interruption:
            finder.camera.cameraError = .cameraInterrupted
            try finder.camera.sendFrame(rotation: 90)
            await next(&actions) { if case .failed(1, .cameraInterrupted) = $0 { true } else { false } }
            #expect(finder.store.state.phase == .idle && !finder.store.state.releasePending)
        }
        finder.permission.granted = false
        await finder.runner.run(.notify(1)) { _ in Issue.record("A notification sends no action") }
        #expect(finder.camera.notifications == 0)

        await #expect(throws: ObjectFinderReleaseFailure(failure: .operation("release failed"))) {
            try await finder.release()
        }
        #expect(finder.store.state.phase == .idle && finder.store.state.releasePending)
        #expect(finder.store.state.failure == .operation("release failed"))
        finder.permission.granted = true
        finder.store.send(.appeared)
        finder.store.send(.start)
        finder.store.send(.analyzed(1, frame))
        await finder.runner.run(.notify(1)) { _ in }
        #expect(
            finder.store.state.phase == .idle && finder.store.state.frame == nil && finder.store.state.matches.isEmpty)
        #expect(finder.camera.authorizations == 1 && finder.camera.notifications == 0)

        finder.camera.releaseError = nil
        finder.camera.blocksRelease = true
        var releasing = finder.camera.releasing.stream.makeAsyncIterator()
        let retry = Task { try await finder.release() }
        retry.cancel()
        _ = await releasing.next()
        finder.store.send(.start)
        #expect(finder.store.state.phase == .stopping && finder.camera.authorizations == 1)
        finder.camera.releaseFinder()
        try await retry.value
        #expect(finder.store.state.phase == .idle && !finder.store.state.releasePending)
        #expect(finder.store.state.failure == nil && finder.camera.frames == nil)
        let closes = await finder.encoder.closeCancellation
        #expect(closes == Array(repeating: false, count: exit == .interruption ? 3 : 2))

        finder.camera.cameraError = nil
        finder.store.send(.start)
        _ = await requests.next()
        #expect(finder.store.state.phase == .searching && finder.store.state.session == 2)
        #expect(await finder.encoder.loads == [selected, selected])
        try await finder.release()
    }

    @Test(
        "CAMERA-04 Search text and settings are retained, live frames and boxes do not survive a stop, and rotation, expiry and version checks still apply"
    )
    func retainedInputs() async throws {
        let finder = try ManagedFinder(folder: selected)
        var requests = finder.requests.makeAsyncIterator()
        finder.store.send(.queryChanged("glasses"))
        finder.store.send(.thresholdChanged(0.72))
        finder.store.send(.start)
        _ = await requests.next()
        let frame = ObjectFinderFrame(candidates: [candidate], seconds: 0.1, captureGeneration: 3)
        finder.store.send(.analyzed(1, frame))
        var preview = ObjectPreviewGate()
        preview.expect(finder.store.state.frame?.captureGeneration)
        preview.receive([candidate], generation: 3, validUntil: 12)
        #expect(finder.store.state.matches == [candidate] && preview.visible(at: 10) == [candidate])
        #expect(preview.visible(at: 12.5).isEmpty)
        finder.store.send(.poseChanged(.landscapeLeft))
        finder.store.send(.analyzed(1, frame))
        #expect(finder.store.state.frame == nil && finder.store.state.orientationRevision == 1)
        finder.store.send(.poseChanged(.portrait))
        finder.store.send(.analyzed(1, frame))
        #expect(finder.store.state.frame == nil)
        finder.store.send(.queryChanged("keys"))

        try await finder.release()
        let stopped = finder.store.state
        #expect(stopped.query == "keys" && stopped.submittedQuery == "glasses" && stopped.minimumSimilarity == 0.72)
        #expect(stopped.phase == .idle && stopped.frame == nil && stopped.matches.isEmpty)
        preview.expect(stopped.phase == .searching ? stopped.frame?.captureGeneration : nil)
        #expect(preview.visible(at: 10).isEmpty)

        let replaced = try ManagedFinder(folder: replacement, retaining: stopped)
        var replacedRequests = replaced.requests.makeAsyncIterator()
        var expected = ObjectFinderState(folder: replacement, execution: .managed)
        expected.query = "keys"
        expected.submittedQuery = "glasses"
        expected.minimumSimilarity = 0.72
        expected.session = 1
        #expect(replaced.store.state == expected)
        replaced.store.send(.appeared)
        replaced.store.send(.analyzed(1, frame))
        #expect(replaced.store.state.frame == nil && replaced.camera.authorizations == 0)
        #expect(replaced.permission.checks == 0)
        replaced.store.send(.start)
        _ = await replacedRequests.next()
        replaced.store.send(.analyzed(1, frame))
        #expect(replaced.store.state.session == 2 && replaced.store.state.frame == nil)
        #expect(await replaced.encoder.loads == [replacement])
        #expect(await replaced.encoder.queries == ["keys"])
        #expect(await finder.encoder.loads == [selected])
        try await replaced.release()
    }
}
