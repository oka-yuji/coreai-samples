import ArgumentParser
@_spi(Verification) import CoreAIBackend
import Darwin
import Foundation
import Testing

@testable import coreai_verify

@Suite(.timeLimit(.minutes(1))) struct MeasurementRecordTests {
    private static let byteKeys = ["rss_bytes", "footprint_bytes", "graphics_footprint_bytes"]

    private func json(_ value: some Encodable) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: JSONEncoder.records.encode(value)) as? [String: Any])
    }

    private func chat(runtime: String, _ extra: [String] = []) -> [String] {
        ["--model", "/not-loaded", "--runtime", runtime, "--prompt", "a"] + extra
    }

    private func speedRun(peakRssBytes: UInt64?, decodeTokS: Double?) -> SpeedRecord {
        SpeedRecord(
            engine: "own", round: 1, loadS: 1, warmupS: 1, specializeS: nil, cached: nil, promptTokens: 4,
            prefillS: 1, prefillTokS: 4, maxTokens: 1, generatedTokens: 1, stopReason: "maxTokens", decodeS: 0,
            decodeTokS: decodeTokS, requestedCompute: nil, expectFrequentReshapes: nil, peakRssBytes: peakRssBytes,
            rssBytes: nil)
    }

    @Test("A failed memory capture is recorded with its status and error and without byte values")
    func failedMemory() throws {
        let memory = try json(Memory.failed(code: KERN_INVALID_ARGUMENT))
        #expect(memory["status"] as? String == "failed")
        #expect(memory["error"] as? String == "task_info returned \(KERN_INVALID_ARGUMENT)")
        #expect(memory["unit"] as? String == "B")
        #expect(memory["captured_at"] is String)
        #expect(Self.byteKeys.allSatisfy { memory[$0] == nil })
    }

    @Test("A successful memory capture is recorded with its status, unit, ISO 8601 time and byte values")
    func capturedMemory() throws {
        let memory = try json(Memory.now())
        #expect(memory["status"] as? String == "ok")
        #expect(memory["error"] == nil)
        #expect(memory["unit"] as? String == "B")
        let time = try #require(memory["captured_at"] as? String)
        #expect(ISO8601DateFormatter().date(from: time) != nil)
        #expect(Self.byteKeys.allSatisfy { memory[$0] is Int })
    }

    @Test("Speed runs without a peak RSS or a decode rate are counted out of the summary instead of becoming zero")
    func summaryWithoutObservations() throws {
        let unobserved = speedRun(peakRssBytes: nil, decodeTokS: nil)
        for runs in [[], [unobserved, unobserved]] {
            let summary = try json(SpeedSummary(engine: "own", runs: runs, own: runs))
            #expect(summary["peak_rss_bytes"] == nil && summary["peak_rss_runs"] as? Int == 0)
            #expect(summary["decode_tok_s"] == nil && summary["decode_runs"] as? Int == 0)
            #expect(summary["own_decode_ratio"] == nil)
        }
        let observed = speedRun(peakRssBytes: 100, decodeTokS: 10)
        let summary = try json(SpeedSummary(engine: "own", runs: [observed, unobserved], own: [observed]))
        #expect(summary["peak_rss_bytes"] as? Int == 100 && summary["peak_rss_runs"] as? Int == 1)
        #expect(summary["decode_tok_s"] as? Double == 10 && summary["decode_runs"] as? Int == 1)
    }

    @Test("A speed record survives the round trip from the child process to the parent")
    func speedRecordRoundTrip() throws {
        let record = speedRun(peakRssBytes: 100, decodeTokS: 10)
        let decoded = try JSONDecoder.snake.decode(SpeedRecord.self, from: JSONEncoder.records.encode(record))
        #expect(try JSONEncoder.records.encode(decoded) == JSONEncoder.records.encode(record))
        let fields = try json(record)
        #expect(fields["max_tokens"] as? Int == 1 && fields["decode_tokens"] == nil)
        #expect(fields["observed_compute"] as? String == "unknown")
    }

    @Test("The decode rate needs two generated tokens and an own run's stop reason compares the count with the limit")
    func speedComputations() {
        #expect(SpeedRecord.decodeRate(generated: 0, seconds: 1) == nil)
        #expect(SpeedRecord.decodeRate(generated: 1, seconds: 1) == nil)
        #expect(SpeedRecord.decodeRate(generated: 11, seconds: 2) == 5)
        #expect(SpeedRecord.ownStopReason(generated: 32, maxTokens: 32) == "maxTokens")
        #expect(SpeedRecord.ownStopReason(generated: 5, maxTokens: 32) == "stopToken")
    }

    @Test("An official run records the reason its iterator ended, so a limit shortened by the context stays maxTokens")
    func officialStopReasons() {
        #expect(SpeedRecord.officialStopReason(.maxTokens) == "maxTokens")
        #expect(SpeedRecord.officialStopReason(.eos) == "stopToken")
        #expect(SpeedRecord.officialStopReason(.stopSequence("</s>")) == "stopSequence")
        #expect(SpeedRecord.officialStopReason(.cancelled) == "cancelled")
        #expect(SpeedRecord.officialStopReason(.error) == "error")
        #expect(SpeedRecord.officialStopReason(nil) == "unknown")
    }

    @Test(
        "Measurement is off by default and each level is accepted for the own runtime",
        arguments: MeasureChoice.allCases)
    func measureOption(_ choice: MeasureChoice) throws {
        #expect(try Chat.parse(chat(runtime: "own")).experiment.settings.measurement == .off)
        let parsed = try Chat.parse(chat(runtime: "own", ["--measure", choice.rawValue]))
        #expect(parsed.experiment.settings.measurement.rawValue == choice.rawValue)
    }

    @Test("The official runtime and the tools command reject the measure option before loading")
    func measureRejected() throws {
        let options = try ExperimentOptions.parse(["--measure", "trace"])
        #expect(throws: ValidationError.self) { try options.validate(.official) }
        #expect(throws: (any Error).self) { try Chat.parse(chat(runtime: "official", ["--measure", "summary"])) }
        #expect(throws: (any Error).self) {
            try ToolRoundTrip.parse(["--model", "/not-loaded", "--runtime", "own", "--measure", "trace"])
        }
    }
}
