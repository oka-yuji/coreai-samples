import Foundation
import ObjectFinderCore
import ObjectFinderFeature

@MainActor
public final class LiveObjectFinderService: ObjectFinderService {
    private let devices: ObjectFinderDevices
    private var busy = false
    private var stopRequested = false
    private var currentSession = 0
    private var stoppedSession = 0
    private var retainOnStop = false
    private var retainedFinder = false
    private var cleanupFailure: (any Error)?
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private var configuration = ObjectCameraConfiguration()

    init(devices: ObjectFinderDevices) { self.devices = devices }

    public func search(
        folder: URL, query: String, minimumSimilarity: Double, rotation: ObjectCameraRotation, started: () -> Void,
        receive: (ObjectFinderFrame) -> Void
    ) async throws {
        try Task.checkCancellation()
        guard !busy else { throw ObjectFinderError.busy }
        guard rotation.session > stoppedSession else { throw CancellationError() }
        currentSession = rotation.session
        busy = true
        stopRequested = false
        retainOnStop = false
        retainedFinder = false
        cleanupFailure = nil
        configuration.update(rotation)
        devices.beginOrientation()
        var failure: (any Error)?
        var reusable = false
        do {
            try await devices.authorize()
            try checkStop()
            var rotation = configuration.rotation
            var frames = try await devices.startCamera(rotation.angle)
            var iterator = frames.makeAsyncIterator()
            try checkStop()
            try await devices.prepare(folder, query)
            reusable = true
            try checkStop()
            started()
            while true {
                try checkStop()
                if rotation != configuration.rotation {
                    await devices.stopCamera()
                    try checkStop()
                    rotation = configuration.rotation
                    frames = try await devices.startCamera(rotation.angle)
                    iterator = frames.makeAsyncIterator()
                }
                try await devices.requestFrame()
                try checkStop()
                guard let frame = try await iterator.next() else { break }
                try checkStop()
                try await devices.checkCamera()
                guard rotation == configuration.rotation, frame.rotation == rotation.angle else { continue }
                let start = ContinuousClock.now
                reusable = false
                let candidates = try await devices.candidates(frame.image)
                reusable = true
                try checkStop()
                try await devices.checkCamera()
                guard rotation == configuration.rotation, frame.rotation == rotation.angle else { continue }
                let matches = ObjectCandidateSelection.matching(candidates, minimumSimilarity: minimumSimilarity)
                let tracked = await devices.trackCandidates(frame, matches)
                try checkStop()
                try await devices.checkCamera()
                guard rotation == configuration.rotation else { continue }
                let duration = (ContinuousClock.now - start).components
                receive(
                    ObjectFinderFrame(
                        candidates: tracked,
                        seconds: Double(duration.seconds) + Double(duration.attoseconds) / 1e18,
                        rotation: frame.rotation, orientationRevision: rotation.revision,
                        captureGeneration: frame.generation))
            }
            try checkStop()
        } catch {
            failure = error
        }
        await withTaskCancellationShield {
            await devices.stopCamera()
            if stopRequested, retainOnStop, reusable, failure is CancellationError {
                retainedFinder = true
            } else {
                do { try await devices.closeFinder() } catch {
                    cleanupFailure = error
                    failure = failure ?? error
                }
            }
            devices.endOrientation()
            finish()
        }
        if let failure { throw failure }
    }

    public func stop(session: Int, retainingModel: Bool) async throws {
        guard session >= currentSession else { return }
        currentSession = session
        stoppedSession = max(stoppedSession, session)
        if !stopRequested { retainOnStop = retainingModel }
        if !retainingModel { retainOnStop = false }
        stopRequested = true
        await withTaskCancellationShield { await devices.stopCamera() }
        guard session == currentSession else { return }
        if busy {
            await withCheckedContinuation { waiters.append($0) }
        }
        guard session == currentSession else { return }
        if let cleanupFailure { throw cleanupFailure }
        guard !retainingModel, retainedFinder else { return }
        busy = true
        retainedFinder = false
        defer { finish() }
        do {
            try await withTaskCancellationShield { try await devices.closeFinder() }
        } catch {
            cleanupFailure = error
            throw error
        }
    }

    public func notify() {
        guard busy, !stopRequested else { return }
        devices.notify()
    }

    public func setRotation(_ rotation: ObjectCameraRotation) {
        configuration.update(rotation)
    }

    private func checkStop() throws {
        try Task.checkCancellation()
        if stopRequested { throw CancellationError() }
    }

    private func finish() {
        busy = false
        let pending = waiters
        waiters = []
        for waiter in pending { waiter.resume() }
    }
}
