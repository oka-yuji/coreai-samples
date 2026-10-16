import CoreGraphics
import Foundation
import ObjectFinderCore
import ObjectFinderFeature
import Testing

@testable import ObjectFinderRuntime

@MainActor
struct LiveObjectFinderServiceTests {
    private let folder = URL(filePath: "/model")
    private let initial = ObjectCameraRotation(session: 1, revision: 0, angle: 90)

    @Test("Denied camera permission releases resources without starting a model or camera")
    func deniedPermission() async {
        let fake = FinderDevicesFake()
        fake.authorizationError = .cameraDenied
        let service = LiveObjectFinderService(devices: fake.devices)
        await #expect(throws: ObjectFinderFailure.cameraDenied) {
            try await service.search(
                folder: folder, query: "glasses", rotation: initial, started: {}, receive: { _ in })
        }
        #expect(fake.prepares == 0 && fake.angles.isEmpty && fake.closes == 1 && fake.orientationEnds == 1)
    }

    @Test("Stopping during model preparation waits for cleanup and never opens the camera", .timeLimit(.minutes(1)))
    func stopDuringPreparation() async throws {
        let fake = FinderDevicesFake()
        fake.blocksPreparation = true
        let service = LiveObjectFinderService(devices: fake.devices)
        let search = Task {
            try await service.search(
                folder: folder, query: "glasses", rotation: initial, started: {}, receive: { _ in })
        }
        var entered = fake.preparing.stream.makeAsyncIterator()
        _ = await entered.next()
        let stop = Task { await service.stop() }
        var stopping = fake.stopping.stream.makeAsyncIterator()
        _ = await stopping.next()
        #expect(fake.closes == 0)
        await #expect(throws: ObjectFinderError.busy) {
            try await service.search(folder: folder, query: "keys", rotation: initial, started: {}, receive: { _ in })
        }
        fake.releasePreparation()
        await #expect(throws: CancellationError.self) { try await search.value }
        await stop.value
        #expect(fake.angles.isEmpty && fake.closes == 1 && fake.orientationEnds == 1)
    }

    @Test(
        "Rotation during inference drops the old frame and restarts capture before accepting new results",
        .timeLimit(.minutes(1)))
    func rotateDuringInference() async throws {
        let fake = FinderDevicesFake()
        fake.blocksInference = true
        let service = LiveObjectFinderService(devices: fake.devices)
        let received = AsyncStream<ObjectFinderFrame>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let search = Task {
            try await service.search(
                folder: folder, query: "glasses", rotation: initial, started: {},
                receive: { received.continuation.yield($0) })
        }
        var starts = fake.started.stream.makeAsyncIterator()
        _ = await starts.next()
        try fake.sendFrame(rotation: 90)
        var entered = fake.inferring.stream.makeAsyncIterator()
        _ = await entered.next()
        service.setRotation(.init(session: 1, revision: 1, angle: 0))
        fake.releaseInference()
        _ = await starts.next()
        #expect(fake.angles == [90, 0] && fake.inferences == 1)
        try fake.sendFrame(rotation: 0)
        var output = received.stream.makeAsyncIterator()
        let frame = await output.next()
        #expect(frame?.rotation == 0 && frame?.orientationRevision == 1)
        search.cancel()
        _ = await search.result
        #expect(fake.closes == 1 && fake.orientationEnds == 1)
        received.continuation.finish()
    }

    @Test("Camera interruption discards in-flight results and buffered frames before cleanup", .timeLimit(.minutes(1)))
    func interruptionDuringInference() async throws {
        let fake = FinderDevicesFake()
        fake.blocksInference = true
        let service = LiveObjectFinderService(devices: fake.devices)
        var frames: [ObjectFinderFrame] = []
        let search = Task {
            try await service.search(
                folder: folder, query: "glasses", rotation: initial, started: {}, receive: { frames.append($0) })
        }
        var starts = fake.started.stream.makeAsyncIterator()
        _ = await starts.next()
        try fake.sendFrame(rotation: 90)
        var entered = fake.inferring.stream.makeAsyncIterator()
        _ = await entered.next()
        try fake.sendFrame(rotation: 90)
        fake.cameraError = .cameraInterrupted
        fake.releaseInference()
        await #expect(throws: ObjectFinderFailure.cameraInterrupted) { try await search.value }
        #expect(frames.isEmpty && fake.inferences == 1 && fake.closes == 1)
    }
}

@MainActor
private final class FinderDevicesFake {
    let preparing = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
    let stopping = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
    let started = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
    let inferring = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
    var authorizationError: ObjectFinderFailure?
    var cameraError: ObjectFinderFailure?
    var blocksPreparation = false
    var blocksInference = false
    var prepares = 0
    var inferences = 0
    var closes = 0
    var orientationEnds = 0
    var angles: [Double] = []
    var frames: AsyncThrowingStream<CapturedObjectFrame, any Error>.Continuation?
    private var preparation: CheckedContinuation<Void, Never>?
    private var inference: CheckedContinuation<Void, Never>?

    var devices: ObjectFinderDevices {
        ObjectFinderDevices(
            authorize: { if let error = self.authorizationError { throw error } },
            prepare: { _, _ in
                self.prepares += 1
                if self.blocksPreparation {
                    await withCheckedContinuation {
                        self.preparation = $0
                        self.preparing.continuation.yield(())
                    }
                }
            },
            startCamera: { angle in
                self.angles.append(angle)
                let channel = AsyncThrowingStream<CapturedObjectFrame, any Error>.makeStream(
                    bufferingPolicy: .bufferingNewest(1))
                self.frames = channel.continuation
                self.started.continuation.yield(())
                return channel.stream
            },
            stopCamera: {
                self.frames?.finish()
                self.frames = nil
                self.stopping.continuation.yield(())
            },
            checkCamera: { if let error = self.cameraError { throw error } },
            candidates: { _ in
                self.inferences += 1
                if self.blocksInference {
                    await withCheckedContinuation {
                        self.inference = $0
                        self.inferring.continuation.yield(())
                    }
                }
                return []
            },
            closeFinder: { self.closes += 1 },
            beginOrientation: {}, endOrientation: { self.orientationEnds += 1 }, notify: {})
    }

    func sendFrame(rotation: Double) throws {
        let context = try #require(
            CGContext(
                data: nil, width: 2, height: 2, bitsPerComponent: 8, bytesPerRow: 8,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let image = try #require(context.makeImage())
        frames?.yield(CapturedObjectFrame(image: image, jpeg: Data([1]), rotation: rotation))
    }

    func releasePreparation() {
        blocksPreparation = false
        preparation?.resume()
        preparation = nil
    }

    func releaseInference() {
        blocksInference = false
        inference?.resume()
        inference = nil
    }
}
