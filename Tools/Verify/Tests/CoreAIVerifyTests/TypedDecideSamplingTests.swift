import ArgumentParser
import Foundation
import Synchronization
import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct TypedDecideSamplingTests {
    private let required = ["--bundle", "/not-loaded", "--input", "/in.json", "--output", "/out.json"]

    @Test("Cache clearing and in-run memory sampling are off by default and set by their flags")
    func flags() throws {
        let plain = try TypedDecide.parse(required)
        #expect(!plain.clearCache && !plain.sampleMemory)
        let measured = try TypedDecide.parse(required + ["--clear-cache", "--sample-memory"])
        #expect(measured.clearCache && measured.sampleMemory)
    }

    @Test(
        "The sampler keeps the largest RSS and footprint every 100 ms during the run, counts failed samples and adds one after it"
    )
    func largestDuringTheRun() async throws {
        let values: [UInt64?] = [100, nil, 300, 200]
        let captured = Mutex(0)
        let finished = Mutex(false)
        let afterRun = Mutex<[Bool]>([])
        let deadlines = Mutex<[ContinuousClock.Instant]>([])
        let (signal, continuation) = AsyncStream.makeStream(of: Int.self)
        let started = ContinuousClock.now
        let wait: @Sendable (ContinuousClock.Instant) async throws -> Void = { deadline in
            deadlines.withLock { $0.append(deadline) }
            try Task.checkCancellation()
            await Task.yield()
        }
        let capture: @Sendable () -> Memory = {
            let index = captured.withLock { count in
                count += 1
                return count - 1
            }
            continuation.yield(index)
            let done = finished.withLock { $0 }
            afterRun.withLock { $0.append(done) }
            guard let rss = values[min(index, values.count - 1)] else { return Memory.failed(code: 5) }
            return Memory(rssBytes: rss, footprintBytes: rss / 2, graphicsFootprintBytes: 0)
        }
        let result = try await sampleResident(started: started, waitUntil: wait, capture: capture) {
            for await index in signal where index >= values.count - 1 { break }
            finished.withLock { $0 = true }
            return 7
        }
        let total = captured.withLock { $0 }
        #expect(result.value == 7)
        #expect(result.resident.rssBytes == 300 && result.resident.footprintBytes == 150)
        #expect(result.resident.samples + result.resident.failedSamples == total)
        #expect(result.resident.failedSamples == 1)
        #expect(result.resident.intervalSeconds == 0.1)
        #expect(afterRun.withLock { $0.last } == true)
        let recorded = deadlines.withLock { $0 }
        #expect(!recorded.isEmpty)
        for (index, deadline) in recorded.enumerated() {
            #expect(deadline == started + .milliseconds(100 * (index + 1)))
        }
    }

    @Test("A sampled row encodes the largest RSS and footprint, and an unsampled row has no sampling key")
    func encoding() throws {
        var row = TypedDecideRecord(
            id: "a", kind: .noul, tokenCount: 3, tokensSha256: tokensDigest([1, 2, 3]), capacity: 256,
            logits: Array(repeating: 0, count: 24), probabilities: [0.5, 0.5], probabilitiesT1: [0.5, 0.5], top: 0,
            elapsedSeconds: 0.5)
        let plain = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder.records.encode(row)) as? [String: Any])
        #expect(plain["resident_bytes_max"] == nil)
        row.residentBytesMax = ResidentMaximum(rssBytes: 9, footprintBytes: 4, samples: 3, failedSamples: 0)
        let sampled = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder.records.encode(row)) as? [String: Any])
        let maximum = try #require(sampled["resident_bytes_max"] as? [String: Any])
        let expected: NSDictionary = [
            "rss_bytes": 9, "footprint_bytes": 4, "samples": 3, "failed_samples": 0, "interval_seconds": 0.1,
        ]
        #expect(maximum as NSDictionary == expected)
    }
}
