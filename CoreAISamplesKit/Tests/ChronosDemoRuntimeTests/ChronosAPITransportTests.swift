import ChronosFeature
import Foundation
import Testing

@testable import ChronosDemoRuntime

@Suite(.timeLimit(.minutes(1)))
struct ChronosAPITransportTests {
    @Test("A1 The production receiver cancels during reception at the byte limit")
    func streamingLimit() async throws {
        let tasks = AsyncStream<URLSessionDataTask>.makeStream()
        let receiver = NYCHTTPTransport(limit: 8, start: { tasks.continuation.yield($0) })
        let request = URLRequest(url: try NYCAPIContract.catalogURL())
        let operation = Task { try await receiver.receive(request) }
        var iterator = tasks.stream.makeAsyncIterator()
        let task = try #require(await iterator.next())
        let session = URLSession(configuration: .ephemeral)
        defer {
            session.invalidateAndCancel()
            tasks.continuation.finish()
        }
        let response = try #require(
            HTTPURLResponse(url: NYCAPIContract.catalogURL(), statusCode: 200, httpVersion: nil, headerFields: nil))
        receiver.urlSession(session, dataTask: task, didReceive: response) { #expect($0 == .allow) }
        receiver.urlSession(session, dataTask: task, didReceive: Data(repeating: 1, count: 8))
        receiver.urlSession(session, dataTask: task, didReceive: Data([2]))
        await #expect(throws: ChronosDemoFailure.responseTooLarge) { try await operation.value }
        #expect(task.state == .canceling || task.state == .completed)
        receiver.urlSession(session, task: task, didCompleteWithError: nil)
    }

    @Test("A1 Redirects and reception cancellation terminate the production receiver", arguments: [false, true])
    func receptionStops(_ redirect: Bool) async throws {
        let tasks = AsyncStream<URLSessionDataTask>.makeStream()
        let receiver = NYCHTTPTransport(start: { tasks.continuation.yield($0) })
        let request = URLRequest(url: try NYCAPIContract.catalogURL())
        let operation = Task { try await receiver.receive(request) }
        var iterator = tasks.stream.makeAsyncIterator()
        let task = try #require(await iterator.next())
        let session = URLSession(configuration: .ephemeral)
        defer {
            session.invalidateAndCancel()
            tasks.continuation.finish()
        }
        if redirect {
            let response = try #require(
                HTTPURLResponse(url: NYCAPIContract.catalogURL(), statusCode: 302, httpVersion: nil, headerFields: nil))
            receiver.urlSession(session, task: task, willPerformHTTPRedirection: response, newRequest: request) {
                #expect($0 == nil)
            }
            await #expect(throws: ChronosDemoFailure.redirectDenied) { try await operation.value }
        } else {
            operation.cancel()
            await #expect(throws: CancellationError.self) { try await operation.value }
        }
        receiver.urlSession(session, dataTask: task, didReceive: Data([1, 2, 3]))
        receiver.urlSession(session, task: task, didCompleteWithError: URLError(.cancelled))
    }

    @Test("A3 A newer coverage refresh wins even when the old transport ignores cancellation")
    func refreshGeneration() async throws {
        let fake = try ChronosAPITransportFake()
        let client = ChronosAPIFixture.client(fake)
        await fake.hold()
        let old = Task { try await client.coverage(refresh: true) }
        var events = fake.started.stream.makeAsyncIterator()
        _ = await events.next()
        await fake.allowNew()
        _ = try await client.coverage(refresh: true)
        await fake.finish()
        await #expect(throws: CancellationError.self) { try await old.value }
        let calls = await fake.requests.count
        _ = try await client.coverage()
        #expect(await fake.requests.count == calls)
    }

    @Test("A3 Cancellation of a noncooperative chunk stops every subsequent chunk")
    func cancelledChunk() async throws {
        let fake = try ChronosAPITransportFake()
        let client = ChronosAPIFixture.client(fake)
        _ = try await client.coverage()
        await fake.hold(path: "/resource/")
        let operation = Task {
            try await client.window(
                start: ChronosAPIFixture.date("2025-01-03T22:00:00Z"), count: 1008, areaIDs: ["230"])
        }
        var events = fake.started.stream.makeAsyncIterator()
        _ = await events.next()
        _ = await events.next()
        _ = await events.next()
        operation.cancel()
        await fake.finish()
        await #expect(throws: CancellationError.self) { try await operation.value }
        #expect(await fake.requests.count == 3)
    }
}
