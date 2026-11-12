import ChronosRuntime
import Foundation
import Testing

@testable import ForecastFeature
@testable import ForecastRuntime

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct ForecastBoundaryTests {
    @Test("S2 A forged context, actual, baseline or series is rejected before the runtime factory runs")
    func authorizedSlice() async throws {
        let root = try ForecastFixture.directory()
        defer { try? FileManager.default.removeItem(at: root) }
        var creations = 0
        let service = LiveForecastService(
            resource: ForecastFixture.session(1).resource, repository: ForecastFixture.repository(root),
            makeRuntime: { _, _ in
                creations += 1
                return ChronosRuntime(factory: { ForecastPredictorFake(identity: ForecastFixture.identity()) })
            })
        let dataset = try await service.load(refresh: false)
        let request = try dataset.request(seriesID: "B1_1", originIndex: 365, horizon: 14)
        for field in ["context", "actuals", "baseline", "series", "dates"] {
            let forged = ForecastRequest(
                kind: request.kind, seriesID: field == "series" ? "unknown" : request.seriesID,
                originIndex: request.originIndex, horizon: request.horizon,
                timestamps: field == "dates" ? [] : request.timestamps,
                context: field == "context" ? Array(repeating: 9000, count: 365) : request.context,
                actuals: field == "actuals" ? Array(repeating: 9000, count: 14) : request.actuals,
                baseline: field == "baseline" ? Array(repeating: 9000, count: 14) : request.baseline)
            await #expect(throws: ForecastFailure.invalidRequest) { try await service.predict(forged) }
        }
        #expect(creations == 0)
        await service.release()
    }

    @Test("S2 Repository cancellation rejects noncooperative bytes and leaves the prior cache intact")
    func cancellation() async throws {
        let root = try ForecastFixture.directory()
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try await ForecastFixture.repository(root).load()
        let cache = root.appending(path: "sales.csv")
        let before = try Data(contentsOf: cache)
        let fetch = ForecastFetchGate()
        var starts = fetch.starts.stream.makeAsyncIterator()
        let repository = SalesDataRepository(
            cacheDirectory: root, fetch: { await fetch.read() }, admit: SalesDataset.parse)
        let operation = Task { try await repository.load(refresh: true) }
        _ = await starts.next()
        await #expect(throws: ForecastFailure.busy) { try await repository.load() }
        operation.cancel()
        await fetch.finish()
        do {
            _ = try await operation.value
            Issue.record("Cancelled bytes must not publish a dataset")
        } catch {
            #expect(error is CancellationError)
        }
        #expect(try Data(contentsOf: cache) == before)
    }

    @Test("S2 Service release joins a delayed data read and suppresses its completion before accepting a fresh load")
    func serviceRelease() async throws {
        let root = try ForecastFixture.directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let fetch = ForecastFetchGate()
        let service = LiveForecastService(
            repository: SalesDataRepository(
                cacheDirectory: root, fetch: { await fetch.read() }, admit: SalesDataset.parse))
        var starts = fetch.starts.stream.makeAsyncIterator()
        let operation = Task { try await service.load(refresh: false) }
        _ = await starts.next()
        let stops = AsyncStream<Void>.makeStream()
        var started = stops.stream.makeAsyncIterator()
        var released = false
        let release = Task {
            stops.continuation.yield(())
            await service.release()
            released = true
        }
        _ = await started.next()
        #expect(!released)
        await fetch.finish()
        do {
            _ = try await operation.value
            Issue.record("A load completing after release began must be rejected")
        } catch {
            #expect(error is CancellationError)
        }
        await release.value
        #expect(released)
        let dataset = try await service.load(refresh: false)
        #expect(dataset.series.count == 4)
        await service.release()
        stops.continuation.finish()
    }
}

private actor ForecastFetchGate {
    nonisolated let starts = AsyncStream<Void>.makeStream()
    private var continuation: CheckedContinuation<Void, Never>?

    func read() async -> Data {
        await withCheckedContinuation {
            continuation = $0
            starts.continuation.yield(())
        }
        return ForecastFixture.csv()
    }

    func finish() {
        continuation?.resume()
        continuation = nil
    }
}
