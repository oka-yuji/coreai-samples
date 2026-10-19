import FeatureStore
import Foundation
import ObjectFinderCore
import ObjectFinderFeature
import ObjectFinderRuntime
import Testing

struct ObjectFinderFeatureTests {
    private let folder = URL(filePath: "/model")
    private let candidate = ObjectCandidate(
        id: 0, region: .init(x: 0.2, y: 0.2, width: 0.3, height: 0.3), similarity: 0.8)

    private func active() -> ObjectFinderState {
        var state = ObjectFinderState(folder: folder)
        state.query = "glasses"
        _ = ObjectFinderFeature.reduce(&state, .start)
        _ = ObjectFinderFeature.reduce(&state, .started(1))
        return state
    }

    @Test("A precision profile can change only after camera stop is acknowledged")
    func switchProfileAfterStop() {
        var state = active()
        let target = URL(filePath: "/models/embeddinggemma2-int4-int8")
        _ = ObjectFinderFeature.reduce(&state, .folderSelected(target))
        #expect(state.folder == folder)
        _ = ObjectFinderFeature.reduce(&state, .stop)
        _ = ObjectFinderFeature.reduce(&state, .folderSelected(target))
        #expect(state.folder == folder)
        _ = ObjectFinderFeature.reduce(&state, .stopped(state.session))
        _ = ObjectFinderFeature.reduce(&state, .folderSelected(target))
        #expect(state.folder == target && state.phase == .idle)
    }

    @Test("A valid target and model are required and draft edits preserve the active target")
    func startGuards() {
        var state = ObjectFinderState()
        #expect(ObjectFinderFeature.reduce(&state, .start).isEmpty)
        _ = ObjectFinderFeature.reduce(&state, .folderSelected(folder))
        _ = ObjectFinderFeature.reduce(&state, .queryChanged(" \n "))
        #expect(ObjectFinderFeature.reduce(&state, .start).isEmpty)
        _ = ObjectFinderFeature.reduce(&state, .queryChanged(" glasses "))
        #expect(
            ObjectFinderFeature.reduce(&state, .start) == [
                .run(.search(1, folder, "glasses", 0.67, .init(session: 1, revision: 0, angle: 90)), id: .search)
            ])
        _ = ObjectFinderFeature.reduce(&state, .queryChanged("keys"))
        _ = ObjectFinderFeature.reduce(&state, .folderSelected(URL(filePath: "/other")))
        _ = ObjectFinderFeature.reduce(&state, .thresholdChanged(0.3))
        #expect(
            state.query == "keys" && state.submittedQuery == "glasses" && state.folder == folder
                && state.minimumSimilarity == 0.67)
        #expect(ObjectFinderFeature.reduce(&state, .start).isEmpty)
    }

    @Test(
        "Editing a running search changes only the draft until it is applied, and replacement waits for the old operation"
    )
    func applyTypedQuery() {
        var state = active()
        _ = ObjectFinderFeature.reduce(&state, .queryChanged(" red keys "))
        #expect(state.query == " red keys " && state.submittedQuery == "glasses" && state.canApplyQuery)
        #expect(
            ObjectFinderFeature.reduce(&state, .applyQuery) == [
                .cancel(.notification), .run(.stop(1, retainingModel: true), id: .control),
            ])
        #expect(state.phase == .stopping && state.session == 1 && state.canStop)
        _ = ObjectFinderFeature.reduce(&state, .queryChanged("wallet"))
        _ = ObjectFinderFeature.reduce(&state, .analyzed(1, ObjectFinderFrame(candidates: [candidate], seconds: 0.1)))
        #expect(state.frame == nil && state.query == " red keys ")
        #expect(
            ObjectFinderFeature.reduce(&state, .stopped(1)) == [
                .run(.search(2, folder, "red keys", 0.67, .init(session: 2, revision: 0, angle: 90)), id: .search)
            ])
        #expect(state.submittedQuery == "red keys" && state.phase == .starting && !state.restartAfterStop)
    }

    @Test("Stop, backgrounding and dismissal cancel a pending query restart and reject old-session results")
    func cancelQueryRestart() {
        for action in [ObjectFinderAction.stop, .backgrounded, .disappeared] {
            var state = active()
            _ = ObjectFinderFeature.reduce(&state, .queryChanged("keys"))
            _ = ObjectFinderFeature.reduce(&state, .applyQuery)
            #expect(
                ObjectFinderFeature.reduce(&state, action)
                    == [.cancelAll, .run(.stop(1, retainingModel: false), id: .control)])
            #expect(!state.canStop)
            #expect(ObjectFinderFeature.reduce(&state, .stopped(1)).isEmpty)
            _ = ObjectFinderFeature.reduce(&state, .started(1))
            #expect(state.phase == .idle && !state.restartAfterStop && state.frame == nil)
        }
    }

    @Test("A release failure cancels the replacement instead of restarting with uncertain resources")
    func replacementFailure() {
        var state = active()
        _ = ObjectFinderFeature.reduce(&state, .queryChanged("keys"))
        _ = ObjectFinderFeature.reduce(&state, .applyQuery)
        _ = ObjectFinderFeature.reduce(&state, .stopFailed(1, .operation("release failed")))
        #expect(state.phase == .idle && !state.restartAfterStop && state.failure == .operation("release failed"))
        #expect(ObjectFinderFeature.reduce(&state, .stopped(1)).isEmpty)
    }

    @Test("Rotation invalidates live boxes immediately and returning to the same angle rejects old capture updates")
    func livePreviewFollowsFeature() {
        var state = active()
        var preview = ObjectPreviewGate()
        _ = ObjectFinderFeature.reduce(
            &state, .analyzed(1, .init(candidates: [candidate], seconds: 0.1, captureGeneration: 1)))
        preview.expect(state.frame?.captureGeneration)
        preview.receive([candidate], generation: 1, validUntil: 12)
        #expect(preview.visible(at: 10) == [candidate])
        _ = ObjectFinderFeature.reduce(&state, .poseChanged(.landscapeLeft))
        preview.expect(state.frame?.captureGeneration)
        preview.receive([candidate], generation: 1, validUntil: 12)
        #expect(preview.visible(at: 10).isEmpty)
        _ = ObjectFinderFeature.reduce(&state, .poseChanged(.portrait))
        _ = ObjectFinderFeature.reduce(
            &state,
            .analyzed(1, .init(candidates: [candidate], seconds: 0.1, orientationRevision: 2, captureGeneration: 3)))
        preview.expect(state.frame?.captureGeneration)
        preview.receive([candidate], generation: 1, validUntil: 12)
        #expect(preview.visible(at: 10).isEmpty)
        preview.receive([candidate], generation: 3, validUntil: 12)
        #expect(preview.visible(at: 10) == [candidate])
        #expect(preview.visible(at: 13).isEmpty)
    }

    @Test("Stop, close and background transitions reject late results and wait for resource shutdown before restart")
    func stopAndStale() {
        for action in [ObjectFinderAction.stop, .disappeared, .backgrounded] {
            var state = active()
            let frame = ObjectFinderFrame(candidates: [candidate], seconds: 0.3)
            _ = ObjectFinderFeature.reduce(&state, .analyzed(1, frame))
            #expect(
                ObjectFinderFeature.reduce(&state, action) == [
                    .cancelAll, .run(.stop(1, retainingModel: false), id: .control),
                ])
            _ = ObjectFinderFeature.reduce(&state, .analyzed(1, frame))
            _ = ObjectFinderFeature.reduce(&state, .failed(1, .cameraDenied))
            _ = ObjectFinderFeature.reduce(&state, .started(1))
            #expect(state.frame == nil && state.matches.isEmpty && state.failure == nil && state.phase == .stopping)
            #expect(ObjectFinderFeature.reduce(&state, .start).isEmpty)
            _ = ObjectFinderFeature.reduce(&state, .stopped(0))
            #expect(state.phase == .stopping)
            _ = ObjectFinderFeature.reduce(&state, .stopped(1))
            _ = ObjectFinderFeature.reduce(&state, .appeared)
            #expect(
                ObjectFinderFeature.reduce(&state, .start) == [
                    .run(.search(2, folder, "glasses", 0.67, .init(session: 2, revision: 0, angle: 90)), id: .search)
                ])
            _ = ObjectFinderFeature.reduce(&state, .analyzed(1, frame))
            #expect(state.frame == nil)
        }
    }

    @Test("Stable candidates notify once while empty results, permission denial and runtime failures remain distinct")
    func notificationAndErrors() {
        var state = active()
        let frame = ObjectFinderFrame(candidates: [candidate], seconds: 0.2)
        #expect(ObjectFinderFeature.reduce(&state, .analyzed(1, frame)).isEmpty)
        #expect(ObjectFinderFeature.reduce(&state, .analyzed(1, frame)) == [.run(.notify(1), id: .notification)])
        #expect(ObjectFinderFeature.reduce(&state, .analyzed(1, frame)).isEmpty)
        let empty = ObjectFinderFrame(candidates: [], seconds: 0.2)
        for _ in 0..<3 { _ = ObjectFinderFeature.reduce(&state, .analyzed(1, empty)) }
        #expect(state.matches.isEmpty && state.failure == nil && state.phase == .searching && !state.notified)
        _ = ObjectFinderFeature.reduce(&state, .failed(1, .cameraDenied))
        #expect(state.failure == .cameraDenied && state.phase == .idle && state.frame == nil)
        _ = ObjectFinderFeature.reduce(&state, .start)
        #expect(state.failure == nil)
        _ = ObjectFinderFeature.reduce(&state, .failed(2, .operation("failed")))
        #expect(state.failure == .operation("failed"))
    }

    @Test("Rotation clears visible boxes and rejects analysis from the previous orientation")
    func rotationDuringInference() {
        var state = active()
        let portrait = ObjectFinderFrame(candidates: [candidate], seconds: 0.2)
        _ = ObjectFinderFeature.reduce(&state, .analyzed(1, portrait))
        _ = ObjectFinderFeature.reduce(&state, .poseChanged(.landscapeLeft))
        #expect(state.frame == nil && state.matches.isEmpty)
        _ = ObjectFinderFeature.reduce(&state, .analyzed(1, portrait))
        #expect(state.frame == nil)
        let landscape = ObjectFinderFrame(
            candidates: [candidate], seconds: 0.2, rotation: 0, orientationRevision: 1)
        _ = ObjectFinderFeature.reduce(&state, .analyzed(1, landscape))
        _ = ObjectFinderFeature.reduce(&state, .poseChanged(.flat))
        #expect(state.frame == landscape && state.matches == [candidate])
        _ = ObjectFinderFeature.reduce(&state, .poseChanged(.portrait))
        _ = ObjectFinderFeature.reduce(&state, .analyzed(1, portrait))
        #expect(state.frame == nil)
    }

    @Test("A rotation observed during preparation reaches the camera even when the device is then flat")
    func sharedOrientationDuringPreparation() {
        var state = ObjectFinderState(folder: folder)
        state.query = "glasses"
        _ = ObjectFinderFeature.reduce(&state, .start)
        let initial = state.cameraRotation
        let commands = ObjectFinderFeature.reduce(&state, .poseChanged(.landscapeLeft))
        _ = ObjectFinderFeature.reduce(&state, .poseChanged(.flat))
        #expect(commands == [.cancel(.notification), .run(.orient(state.cameraRotation), id: .orientation)])
        var camera = ObjectCameraConfiguration()
        camera.update(state.cameraRotation)
        camera.update(initial)
        #expect(camera.rotation.angle == 0 && camera.rotation.revision == 1)
        _ = ObjectFinderFeature.reduce(&state, .started(1))
        let frame = ObjectFinderFrame(
            candidates: [candidate], seconds: 0.1, rotation: camera.rotation.angle,
            orientationRevision: camera.rotation.revision)
        _ = ObjectFinderFeature.reduce(&state, .analyzed(1, frame))
        #expect(state.frame == frame && state.matches == [candidate])
        camera.update(.init(session: 2, revision: 0, angle: 180))
        camera.update(.init(session: 1, revision: 9, angle: 90))
        #expect(camera.rotation == .init(session: 2, revision: 0, angle: 180))
    }

    @MainActor
    @Test(
        "The runner propagates success and failure and releases camera and inference resources on cancellation",
        .timeLimit(.minutes(1)))
    func runnerAndStore() async throws {
        let service = ControlledFinder()
        let runner = ObjectFinderRunner(service: service)
        let stopped = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        var initial = ObjectFinderState(folder: folder)
        initial.query = "glasses"
        initial.minimumSimilarity = 0.79
        let store = Store<ObjectFinderFeature>(initialState: initial) { effect, send in
            await runner.run(effect) { action in
                send(action)
                if case .stopped = action { stopped.continuation.yield(()) }
            }
        }
        store.send(.start)
        var entered = service.entered.stream.makeAsyncIterator()
        _ = await entered.next()
        #expect(store.state.phase == .searching && service.thresholds == [0.79])
        store.send(.stop)
        var stopping = service.stopping.stream.makeAsyncIterator()
        _ = await stopping.next()
        store.send(.start)
        #expect(service.calls == 1 && store.state.phase == .stopping)
        service.release()
        var finished = stopped.stream.makeAsyncIterator()
        _ = await finished.next()
        #expect(store.state.phase == .idle && store.state.frame == nil && service.closes == 1)
        service.failure = .cameraDenied
        var failure: ObjectFinderFailure?
        await runner.run(.search(4, folder, "keys", 0.67, .init(session: 4, revision: 0, angle: 90))) { action in
            if case .failed(let session, let value) = action {
                #expect(session == 4)
                failure = value
            }
        }
        #expect(failure == .cameraDenied && service.closes == 2)
    }

    @Test("BOX-01 The box limit defaults to three and changes only while stopped and within one to three")
    func boxLimit() {
        var state = ObjectFinderState(folder: folder)
        #expect(state.maximumBoxes == 3)
        #expect(ObjectFinderFeature.reduce(&state, .maximumBoxesChanged(1)).isEmpty && state.maximumBoxes == 1)
        #expect(ObjectFinderFeature.reduce(&state, .maximumBoxesChanged(0)).isEmpty)
        #expect(ObjectFinderFeature.reduce(&state, .maximumBoxesChanged(4)).isEmpty && state.maximumBoxes == 1)
        state.query = "glasses"
        _ = ObjectFinderFeature.reduce(&state, .start)
        #expect(ObjectFinderFeature.reduce(&state, .maximumBoxesChanged(2)).isEmpty && state.maximumBoxes == 1)
        _ = ObjectFinderFeature.reduce(&state, .started(1))
        _ = ObjectFinderFeature.reduce(&state, .stop)
        #expect(ObjectFinderFeature.reduce(&state, .maximumBoxesChanged(3)).isEmpty && state.maximumBoxes == 1)
        _ = ObjectFinderFeature.reduce(&state, .stopped(1))
        #expect(ObjectFinderFeature.reduce(&state, .maximumBoxesChanged(3)).isEmpty && state.maximumBoxes == 3)
    }

    @Test("BOX-02 Selected boxes keep the highest non-overlapping matches up to the box limit")
    func boxLimitSelection() {
        var state = ObjectFinderState(folder: folder)
        _ = ObjectFinderFeature.reduce(&state, .maximumBoxesChanged(2))
        state.query = "glasses"
        _ = ObjectFinderFeature.reduce(&state, .start)
        _ = ObjectFinderFeature.reduce(&state, .started(1))
        let candidates = [
            ObjectCandidate(id: 0, region: .init(x: 0, y: 0, width: 0.2, height: 0.2), similarity: 0.7),
            ObjectCandidate(id: 1, region: .init(x: 0.4, y: 0, width: 0.2, height: 0.2), similarity: 0.9),
            ObjectCandidate(id: 2, region: .init(x: 0, y: 0.5, width: 0.2, height: 0.2), similarity: 0.8),
            ObjectCandidate(id: 3, region: .init(x: 0.5, y: 0.5, width: 0.2, height: 0.2), similarity: 0.5),
        ]
        _ = ObjectFinderFeature.reduce(&state, .analyzed(1, ObjectFinderFrame(candidates: candidates, seconds: 0.1)))
        #expect(state.matches.map(\.id) == [1, 2])
    }

    @Test("The camera sheet is shown only while starting, searching or replacing the query")
    func cameraSheetVisibility() {
        var state = ObjectFinderState(folder: folder)
        state.query = "glasses"
        #expect(!state.showsCamera)
        _ = ObjectFinderFeature.reduce(&state, .start)
        #expect(state.showsCamera)
        _ = ObjectFinderFeature.reduce(&state, .started(1))
        #expect(state.showsCamera)
        _ = ObjectFinderFeature.reduce(&state, .queryChanged("keys"))
        _ = ObjectFinderFeature.reduce(&state, .applyQuery)
        #expect(state.phase == .stopping && state.showsCamera)
        _ = ObjectFinderFeature.reduce(&state, .stopped(1))
        _ = ObjectFinderFeature.reduce(&state, .started(2))
        _ = ObjectFinderFeature.reduce(&state, .stop)
        #expect(state.phase == .stopping && !state.showsCamera)
        _ = ObjectFinderFeature.reduce(&state, .stopped(2))
        #expect(state.phase == .idle && !state.showsCamera)
        _ = ObjectFinderFeature.reduce(&state, .start)
        _ = ObjectFinderFeature.reduce(&state, .failed(3, .cameraDenied))
        #expect(state.phase == .idle && !state.showsCamera)
    }

    @Test("The search log keeps the latest twenty frames newest first and a new search clears it")
    func searchLog() {
        var state = active()
        for index in 0..<25 {
            _ = ObjectFinderFeature.reduce(
                &state,
                .analyzed(1, ObjectFinderFrame(candidates: [], seconds: Double(index), capturedAt: Double(index))))
        }
        #expect(state.log.count == ObjectFinderState.logLimit && state.analyzedFrames == 25)
        #expect(state.log.first?.seconds == 24 && state.log.last?.seconds == 5)
        _ = ObjectFinderFeature.reduce(&state, .stop)
        _ = ObjectFinderFeature.reduce(&state, .stopped(1))
        #expect(state.log.count == ObjectFinderState.logLimit)
        _ = ObjectFinderFeature.reduce(&state, .start)
        #expect(state.log.isEmpty && state.analyzedFrames == 0 && state.peakFootprintBytes == nil)
    }

    @Test("The log reports average analysis time, analyses per second and the peak memory of this search")
    func searchMetrics() {
        var state = active()
        #expect(state.averageSeconds == nil && state.framesPerSecond == nil && state.peakFootprintBytes == nil)
        let samples: [(Double, Double, UInt64)] = [(0.1, 10.0, 300), (0.3, 10.5, 500), (0.2, 11.0, 400)]
        for (seconds, capturedAt, footprint) in samples {
            let memory = ObjectFinderMemory(footprintBytes: footprint, availableBytes: 1_000)
            _ = ObjectFinderFeature.reduce(
                &state,
                .analyzed(
                    1, ObjectFinderFrame(candidates: [], seconds: seconds, capturedAt: capturedAt, memory: memory)))
        }
        #expect(abs((state.averageSeconds ?? 0) - 0.2) < 1e-9)
        #expect(state.framesPerSecond == 2)
        #expect(state.peakFootprintBytes == 500 && state.log.first?.memory?.footprintBytes == 400)
    }
}

@MainActor
private final class ControlledFinder: ObjectFinderService {
    let entered = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
    let stopping = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
    var failure: ObjectFinderFailure?
    var calls = 0
    var thresholds: [Double] = []
    var closes = 0
    var continuation: CheckedContinuation<Void, Never>?
    var waiters: [CheckedContinuation<Void, Never>] = []

    func search(
        folder: URL, query: String, minimumSimilarity: Double, rotation: ObjectCameraRotation, started: () -> Void,
        receive: (ObjectFinderFrame) -> Void
    ) async throws {
        calls += 1
        thresholds.append(minimumSimilarity)
        if let failure {
            closes += 1
            throw failure
        }
        started()
        await withCheckedContinuation {
            continuation = $0
            entered.continuation.yield(())
        }
        receive(ObjectFinderFrame(candidates: [], seconds: 0))
        closes += 1
        let pending = waiters
        waiters = []
        for waiter in pending { waiter.resume() }
        try Task.checkCancellation()
    }

    func stop(session: Int, retainingModel: Bool) async throws {
        if continuation != nil {
            await withCheckedContinuation {
                waiters.append($0)
                stopping.continuation.yield(())
            }
        }
    }

    func notify() {}
    func setRotation(_ rotation: ObjectCameraRotation) {}

    func release() {
        continuation?.resume()
        continuation = nil
    }
}
