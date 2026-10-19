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
                folder: folder, query: "glasses", minimumSimilarity: 0.67, rotation: initial, started: {},
                receive: { _ in })
        }
        #expect(fake.prepares == 0 && fake.angles.isEmpty && fake.closes == 1 && fake.orientationEnds == 1)
    }

    @Test("Stopping during model preparation stops the live camera and waits for cleanup", .timeLimit(.minutes(1)))
    func stopDuringPreparation() async throws {
        let fake = FinderDevicesFake()
        fake.blocksPreparation = true
        let service = LiveObjectFinderService(devices: fake.devices)
        let search = Task {
            try await service.search(
                folder: folder, query: "glasses", minimumSimilarity: 0.67, rotation: initial, started: {},
                receive: { _ in })
        }
        var entered = fake.preparing.stream.makeAsyncIterator()
        _ = await entered.next()
        #expect(fake.angles == [90])
        let stop = Task { try await service.stop(session: 1, retainingModel: false) }
        var stopping = fake.stopping.stream.makeAsyncIterator()
        _ = await stopping.next()
        #expect(fake.closes == 0)
        await #expect(throws: ObjectFinderError.busy) {
            try await service.search(
                folder: folder, query: "keys", minimumSimilarity: 0.67, rotation: initial, started: {},
                receive: { _ in })
        }
        fake.releasePreparation()
        await #expect(throws: CancellationError.self) { try await search.value }
        try await stop.value
        #expect(fake.angles == [90] && fake.frames == nil && fake.closes == 1 && fake.orientationEnds == 1)
    }

    @Test(
        "A slow or suspended model does not block camera startup or the latest-frame input stream",
        .timeLimit(.minutes(1)))
    func previewStartsBeforeModel() async throws {
        let fake = FinderDevicesFake()
        fake.blocksPreparation = true
        let service = LiveObjectFinderService(devices: fake.devices)
        let search = Task {
            try await service.search(
                folder: folder, query: "glasses", minimumSimilarity: 0.67, rotation: initial, started: {},
                receive: { _ in })
        }
        var entered = fake.preparing.stream.makeAsyncIterator()
        _ = await entered.next()
        #expect(fake.angles == [90] && fake.inferences == 0)
        try fake.sendFrame(rotation: 90)
        let yielded = try fake.sendFrame(rotation: 90)
        if case .dropped? = yielded {} else { Issue.record("The camera buffer must keep only the latest frame") }
        search.cancel()
        fake.releasePreparation()
        await #expect(throws: CancellationError.self) { try await search.value }
        #expect(fake.frames == nil && fake.closes == 1)
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
                folder: folder, query: "glasses", minimumSimilarity: 0.67, rotation: initial, started: {},
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
                folder: folder, query: "glasses", minimumSimilarity: 0.67, rotation: initial, started: {},
                receive: { frames.append($0) })
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

    @Test("An interruption during tracking cannot publish a new match", .timeLimit(.minutes(1)))
    func interruptionDuringTracking() async throws {
        let fake = FinderDevicesFake()
        var devices = fake.devices
        devices.trackCandidates = { _, _ in
            fake.cameraError = .cameraInterrupted
            return [ObjectCandidate(id: 0, region: .init(x: 0, y: 0, width: 1, height: 1), similarity: 0.9)]
        }
        let service = LiveObjectFinderService(devices: devices)
        var received = false
        let search = Task {
            try await service.search(
                folder: folder, query: "glasses", minimumSimilarity: 0.67, rotation: initial, started: {},
                receive: { _ in received = true })
        }
        var starts = fake.started.stream.makeAsyncIterator()
        _ = await starts.next()
        try fake.sendFrame(rotation: 90)
        await #expect(throws: ObjectFinderFailure.cameraInterrupted) { try await search.value }
        #expect(!received && fake.closes == 1)
    }

    @Test("A preparing or busy model does not request additional analysis images", .timeLimit(.minutes(1)))
    func imageDemand() async throws {
        let fake = FinderDevicesFake()
        fake.blocksPreparation = true
        fake.blocksInference = true
        let service = LiveObjectFinderService(devices: fake.devices)
        let search = Task {
            try await service.search(
                folder: folder, query: "glasses", minimumSimilarity: 0.67, rotation: initial, started: {},
                receive: { _ in })
        }
        var preparation = fake.preparing.stream.makeAsyncIterator()
        _ = await preparation.next()
        #expect(fake.requests == 0)
        try fake.sendFrame(rotation: 90)
        fake.releasePreparation()
        var inference = fake.inferring.stream.makeAsyncIterator()
        _ = await inference.next()
        #expect(fake.requests == 1 && fake.inferences == 1)
        for _ in 0..<30 { try fake.sendFrame(rotation: 90) }
        #expect(fake.requests == 1)
        search.cancel()
        fake.releaseInference()
        await #expect(throws: CancellationError.self) { try await search.value }
        #expect(fake.requests == 1 && fake.closes == 1)
    }

    @Test(
        "Candidates below the active search threshold never reach tracking, while eligible candidates preserve their IDs and scores",
        .timeLimit(.minutes(1)))
    func trackingThreshold() async throws {
        let fake = FinderDevicesFake()
        let region = ObjectRegion(x: 0, y: 0, width: 1, height: 1)
        let exact = ObjectCandidate(id: 7, region: region, similarity: 0.8)
        let higher = ObjectCandidate(id: 3, region: region, similarity: 0.9)
        fake.candidateResults = [ObjectCandidate(id: 1, region: region, similarity: 0.79), exact, higher]
        let output = AsyncStream<ObjectFinderFrame>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let service = LiveObjectFinderService(devices: fake.devices)
        let search = Task {
            try await service.search(
                folder: folder, query: "glasses", minimumSimilarity: 0.8, rotation: initial, started: {},
                receive: { output.continuation.yield($0) })
        }
        var starts = fake.started.stream.makeAsyncIterator()
        _ = await starts.next()
        try fake.sendFrame(rotation: 90)
        var results = output.stream.makeAsyncIterator()
        let frame = await results.next()
        #expect(fake.trackingInputs == [[exact, higher]] && frame?.candidates == [exact, higher])
        search.cancel()
        _ = await search.result
        output.continuation.finish()
    }

    @Test(
        "Each analyzed frame reports every evaluated similarity, including those below the threshold, and a memory sample",
        .timeLimit(.minutes(1)))
    func frameMetrics() async throws {
        let fake = FinderDevicesFake()
        let region = ObjectRegion(x: 0, y: 0, width: 1, height: 1)
        let match = ObjectCandidate(id: 1, region: region, similarity: 0.8)
        fake.candidateResults = [ObjectCandidate(id: 0, region: region, similarity: 0.4), match]
        let memory = ObjectFinderMemory(footprintBytes: 512, availableBytes: 2_048)
        var devices = fake.devices
        devices.memory = { memory }
        let output = AsyncStream<ObjectFinderFrame>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let service = LiveObjectFinderService(devices: devices)
        let search = Task {
            try await service.search(
                folder: folder, query: "glasses", minimumSimilarity: 0.67, rotation: initial, started: {},
                receive: { output.continuation.yield($0) })
        }
        var starts = fake.started.stream.makeAsyncIterator()
        _ = await starts.next()
        try fake.sendFrame(rotation: 90, capturedAt: 42)
        var results = output.stream.makeAsyncIterator()
        let frame = await results.next()
        #expect(frame?.candidates == [match] && frame?.similarities == [0.4, 0.8])
        #expect(frame?.memory == memory && frame?.capturedAt == 42)
        search.cancel()
        _ = await search.result
        output.continuation.finish()
    }

    @Test("A failed image request stops capture and releases the model before reporting failure")
    func requestFailure() async {
        let fake = FinderDevicesFake()
        var devices = fake.devices
        devices.requestFrame = { throw ObjectFinderFailure.cameraInterrupted }
        let service = LiveObjectFinderService(devices: devices)
        await #expect(throws: ObjectFinderFailure.cameraInterrupted) {
            try await service.search(
                folder: folder, query: "glasses", minimumSimilarity: 0.67, rotation: initial, started: {},
                receive: { _ in Issue.record("No frame was requested successfully") })
        }
        #expect(fake.inferences == 0 && fake.frames == nil && fake.closes == 1)
    }
}

@MainActor
final class FinderDevicesFake {
    let preparing = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
    let stopping = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
    let started = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
    let inferring = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
    let releasing = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
    var authorizationError: ObjectFinderFailure?
    var cameraError: ObjectFinderFailure?
    var blocksPreparation = false
    var blocksInference = false
    var blocksRelease = false
    var releaseError: ObjectFinderFailure?
    var preparationError: ObjectFinderFailure?
    var inferenceError: ObjectFinderFailure?
    var cancelInference = false
    var authorizations = 0
    var notifications = 0
    var prepares = 0
    var requests = 0
    var inferences = 0
    var closes = 0
    var cameraStops = 0
    var closeCancellation: [Bool] = []
    var orientationEnds = 0
    var angles: [Double] = []
    var candidateResults: [ObjectCandidate] = []
    var trackingInputs: [[ObjectCandidate]] = []
    var frames: AsyncThrowingStream<CapturedObjectFrame, any Error>.Continuation?
    private var preparation: CheckedContinuation<Void, Never>?
    private var inference: CheckedContinuation<Void, Never>?
    private var release: CheckedContinuation<Void, Never>?

    var devices: ObjectFinderDevices {
        ObjectFinderDevices(
            authorize: {
                self.authorizations += 1
                if let error = self.authorizationError { throw error }
            },
            prepare: { _, _ in
                self.prepares += 1
                if self.blocksPreparation {
                    await withCheckedContinuation {
                        self.preparation = $0
                        self.preparing.continuation.yield(())
                    }
                }
                if let error = self.preparationError { throw error }
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
                self.cameraStops += 1
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
                if let error = self.inferenceError { throw error }
                if self.cancelInference { throw CancellationError() }
                return self.candidateResults
            },
            closeFinder: {
                self.closes += 1
                self.closeCancellation.append(Task.isCancelled)
                if self.blocksRelease {
                    await withCheckedContinuation {
                        self.release = $0
                        self.releasing.continuation.yield(())
                    }
                }
                if let error = self.releaseError { throw error }
            },
            beginOrientation: {}, endOrientation: { self.orientationEnds += 1 },
            notify: { self.notifications += 1 },
            requestFrame: { self.requests += 1 },
            trackCandidates: { _, candidates in
                self.trackingInputs.append(candidates)
                return candidates
            })
    }

    @discardableResult
    func sendFrame(rotation: Double, capturedAt: Double = 0) throws -> AsyncThrowingStream<
        CapturedObjectFrame, any Error
    >.Continuation
        .YieldResult?
    {
        let context = try #require(
            CGContext(
                data: nil, width: 2, height: 2, bitsPerComponent: 8, bytesPerRow: 8,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let image = try #require(context.makeImage())
        return frames?.yield(CapturedObjectFrame(image: image, rotation: rotation, capturedAt: capturedAt))
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

    func releaseFinder() {
        blocksRelease = false
        release?.resume()
        release = nil
    }
}
