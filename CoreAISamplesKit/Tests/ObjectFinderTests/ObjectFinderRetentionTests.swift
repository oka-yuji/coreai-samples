import Foundation
import ObjectFinderCore
import ObjectFinderFeature
import Testing

@testable import ObjectFinderRuntime

@MainActor
struct ObjectFinderRetentionTests {
    private let folder = URL(filePath: "/model")

    private func start(
        _ service: LiveObjectFinderService, session: Int, ready: AsyncStream<Void>.Continuation
    ) -> Task<Void, any Error> {
        Task {
            try await service.search(
                folder: folder, query: "keys", minimumSimilarity: 0.67,
                rotation: .init(session: session, revision: 0, angle: 90), started: { ready.yield(()) },
                receive: { _ in })
        }
    }

    @Test(
        "Query replacement retains the prepared model only until the next search, and ordinary stop or cancelling replacement releases it before acknowledgment",
        .timeLimit(.minutes(1)))
    func retentionAndRelease() async throws {
        let fake = FinderDevicesFake()
        let service = LiveObjectFinderService(devices: fake.devices)
        let ready = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        var starts = ready.stream.makeAsyncIterator()
        let first = start(service, session: 1, ready: ready.continuation)
        _ = await starts.next()
        try await service.stop(session: 1, retainingModel: true)
        await #expect(throws: CancellationError.self) { try await first.value }
        #expect(fake.closes == 0 && fake.frames == nil)
        let second = start(service, session: 2, ready: ready.continuation)
        _ = await starts.next()
        #expect(fake.prepares == 2 && fake.closes == 0)
        try await service.stop(session: 2, retainingModel: false)
        await #expect(throws: CancellationError.self) { try await second.value }
        #expect(fake.closes == 1 && fake.closeCancellation == [false])
        ready.continuation.finish()
    }

    @Test(
        "Late stops and stopped search sessions cannot affect a newer search, and release in progress rejects overlapping starts",
        .timeLimit(.minutes(1)))
    func releaseGateAndSessions() async throws {
        let fake = FinderDevicesFake()
        let service = LiveObjectFinderService(devices: fake.devices)
        let ready = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        var starts = ready.stream.makeAsyncIterator()
        let first = start(service, session: 1, ready: ready.continuation)
        _ = await starts.next()
        try await service.stop(session: 1, retainingModel: true)
        _ = await first.result
        fake.blocksRelease = true
        let closing = Task { try await service.stop(session: 1, retainingModel: false) }
        var releasing = fake.releasing.stream.makeAsyncIterator()
        _ = await releasing.next()
        await #expect(throws: ObjectFinderError.busy) {
            try await service.search(
                folder: folder, query: "glasses", minimumSimilarity: 0.67,
                rotation: .init(session: 2, revision: 0, angle: 90), started: {}, receive: { _ in })
        }
        fake.releaseFinder()
        try await closing.value
        await #expect(throws: CancellationError.self) {
            try await service.search(
                folder: folder, query: "old", minimumSimilarity: 0.67,
                rotation: .init(session: 1, revision: 0, angle: 90), started: {}, receive: { _ in })
        }
        let second = start(service, session: 2, ready: ready.continuation)
        _ = await starts.next()
        let stopped = fake.cameraStops
        try await service.stop(session: 1, retainingModel: false)
        #expect(fake.cameraStops == stopped && fake.frames != nil)
        try await service.stop(session: 2, retainingModel: false)
        _ = await second.result
        #expect(fake.closes == 2)
        ready.continuation.finish()
    }

    @Test(
        "Cancelling a pending replacement revokes retention while inference is still suspended", .timeLimit(.minutes(1))
    )
    func cancelDuringInference() async throws {
        let fake = FinderDevicesFake()
        fake.blocksInference = true
        let service = LiveObjectFinderService(devices: fake.devices)
        let ready = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        var starts = ready.stream.makeAsyncIterator()
        let search = start(service, session: 1, ready: ready.continuation)
        _ = await starts.next()
        try fake.sendFrame(rotation: 90)
        var inferring = fake.inferring.stream.makeAsyncIterator()
        _ = await inferring.next()
        var stopping = fake.stopping.stream.makeAsyncIterator()
        let replace = Task { try await service.stop(session: 1, retainingModel: true) }
        _ = await stopping.next()
        replace.cancel()
        let stop = Task { try await service.stop(session: 1, retainingModel: false) }
        _ = await stopping.next()
        #expect(fake.closes == 0)
        fake.releaseInference()
        _ = await search.result
        try await replace.value
        try await stop.value
        #expect(fake.closes == 1 && fake.closeCancellation == [false])
        try await service.stop(session: 1, retainingModel: false)
        #expect(fake.closes == 1)
        ready.continuation.finish()
    }

    @Test("A stop before a queued search begins prevents that session from acquiring resources")
    func stopBeforeStart() async throws {
        let fake = FinderDevicesFake()
        let service = LiveObjectFinderService(devices: fake.devices)
        try await service.stop(session: 1, retainingModel: false)
        await #expect(throws: CancellationError.self) {
            try await service.search(
                folder: folder, query: "keys", minimumSimilarity: 0.67,
                rotation: .init(session: 1, revision: 0, angle: 90), started: {}, receive: { _ in })
        }
        #expect(fake.prepares == 0 && fake.angles.isEmpty)
    }

    @Test("Preparation and release failures discard retention and report their failure", .timeLimit(.minutes(1)))
    func failures() async throws {
        let fake = FinderDevicesFake()
        let service = LiveObjectFinderService(devices: fake.devices)
        let ready = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        var starts = ready.stream.makeAsyncIterator()
        let first = start(service, session: 1, ready: ready.continuation)
        _ = await starts.next()
        try await service.stop(session: 1, retainingModel: true)
        _ = await first.result
        fake.preparationError = .operation("prepare failed")
        let second = start(service, session: 2, ready: ready.continuation)
        await #expect(throws: ObjectFinderFailure.operation("prepare failed")) { try await second.value }
        #expect(fake.closes == 1)
        fake.preparationError = nil
        let third = start(service, session: 3, ready: ready.continuation)
        _ = await starts.next()
        try await service.stop(session: 3, retainingModel: true)
        _ = await third.result
        fake.releaseError = .operation("release failed")
        await #expect(throws: ObjectFinderFailure.operation("release failed")) {
            try await service.stop(session: 3, retainingModel: false)
        }
        #expect(fake.closes == 2)
        ready.continuation.finish()
    }

    @Test(
        "A release failure during ordinary shutdown is reported instead of a successful stop", .timeLimit(.minutes(1)))
    func ordinaryReleaseFailure() async throws {
        let fake = FinderDevicesFake()
        fake.releaseError = .operation("release failed")
        let service = LiveObjectFinderService(devices: fake.devices)
        let ready = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        var starts = ready.stream.makeAsyncIterator()
        let search = start(service, session: 1, ready: ready.continuation)
        _ = await starts.next()
        await #expect(throws: ObjectFinderFailure.operation("release failed")) {
            try await service.stop(session: 1, retainingModel: false)
        }
        _ = await search.result
        #expect(fake.closes == 1)
        ready.continuation.finish()
    }

    @Test(
        "An encoder failure discards the model even when replacement requested retention", .timeLimit(.minutes(1)),
        arguments: [false, true])
    func encoderFailure(cancelled: Bool) async throws {
        let fake = FinderDevicesFake()
        fake.blocksInference = true
        fake.cancelInference = cancelled
        fake.inferenceError = cancelled ? nil : .operation("inference failed")
        let service = LiveObjectFinderService(devices: fake.devices)
        let ready = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        var starts = ready.stream.makeAsyncIterator()
        let search = start(service, session: 1, ready: ready.continuation)
        _ = await starts.next()
        try fake.sendFrame(rotation: 90)
        var inferring = fake.inferring.stream.makeAsyncIterator()
        _ = await inferring.next()
        let stop = Task { try await service.stop(session: 1, retainingModel: true) }
        var stopping = fake.stopping.stream.makeAsyncIterator()
        _ = await stopping.next()
        fake.releaseInference()
        _ = await search.result
        try await stop.value
        #expect(fake.closes == 1)
        try await service.stop(session: 1, retainingModel: false)
        #expect(fake.closes == 1)
        ready.continuation.finish()
    }
}
