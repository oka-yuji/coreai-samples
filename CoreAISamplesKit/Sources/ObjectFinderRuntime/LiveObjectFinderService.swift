import Foundation
import ObjectFinderCore
import ObjectFinderFeature

@MainActor
public final class LiveObjectFinderService: ObjectFinderService {
    private let devices: ObjectFinderDevices
    private var busy = false
    private var stopRequested = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private var configuration = ObjectCameraConfiguration()

    init(devices: ObjectFinderDevices) { self.devices = devices }

    public func search(
        folder: URL, query: String, rotation: ObjectCameraRotation, started: () -> Void,
        receive: (ObjectFinderFrame) -> Void
    ) async throws {
        guard !busy else { throw ObjectFinderError.busy }
        busy = true
        stopRequested = false
        configuration.update(rotation)
        devices.beginOrientation()
        var failure: (any Error)?
        do {
            try await devices.authorize()
            try checkStop()
            try await devices.prepare(folder, query)
            try checkStop()
            var rotation = configuration.rotation
            var frames = try await devices.startCamera(rotation.angle)
            var iterator = frames.makeAsyncIterator()
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
                guard let frame = try await iterator.next() else { break }
                try checkStop()
                try await devices.checkCamera()
                guard rotation == configuration.rotation, frame.rotation == rotation.angle else { continue }
                let start = ContinuousClock.now
                let candidates = try await devices.candidates(frame.image)
                try checkStop()
                try await devices.checkCamera()
                guard rotation == configuration.rotation, frame.rotation == rotation.angle else { continue }
                let duration = (ContinuousClock.now - start).components
                receive(
                    ObjectFinderFrame(
                        image: frame.jpeg, candidates: candidates,
                        seconds: Double(duration.seconds) + Double(duration.attoseconds) / 1e18,
                        rotation: frame.rotation, orientationRevision: rotation.revision))
            }
            try checkStop()
        } catch {
            failure = error
        }
        await devices.stopCamera()
        do { try await devices.closeFinder() } catch { failure = failure ?? error }
        devices.endOrientation()
        busy = false
        let pending = waiters
        waiters = []
        for waiter in pending { waiter.resume() }
        if let failure { throw failure }
    }

    public func stop() async {
        stopRequested = true
        await devices.stopCamera()
        if busy {
            await withCheckedContinuation { waiters.append($0) }
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

}
