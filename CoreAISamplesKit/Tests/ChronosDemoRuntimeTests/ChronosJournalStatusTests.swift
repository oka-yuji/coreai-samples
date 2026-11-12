import ChronosFeature
import Foundation
import Testing

@testable import ChronosDemoRuntime

struct ChronosJournalStatusTests {
    @Test(
        "J3 Availability mixtures retain every missing reason without comparing unavailable slots",
        arguments: ["pending", "ambiguous", "unverified", "partial", "complete"])
    func status(kind: String) async throws {
        let record = try await ChronosJournalFixture.record()
        let availability: [SlotAvailability]
        let expected: ChronosCheckStatus
        switch kind {
        case "pending":
            availability = Array(repeating: .unpublished, count: 4)
            expected = .publicationPending
        case "ambiguous":
            availability = [.ambiguousTime, .unpublished, .ambiguousTime, .unpublished]
            expected = .incomparable
        case "unverified":
            availability = [.unverified, .ambiguousTime, .unpublished, .unverified]
            expected = .unverified
        case "partial":
            availability = [.observed, .unpublished, .ambiguousTime, .unverified]
            expected = .partial
        default:
            availability = Array(repeating: .observed, count: 4)
            expected = .compared
        }
        let window = try APIWindow(
            start: record.input.snapshot.origin, count: 4,
            series: record.areas.map { area in
                CountSeries(areaID: area.areaID, values: availability.map { $0 == .observed ? 0 : nil })
            }, availability: availability, datasets: [], fetchedAt: ChronosAPIFixture.now,
            catalog: ChronosAPIFixture.coverage().evidence, metadata: [], responses: [])
        let check = try ChronosActualCheck(
            id: UUID(), record: record, checkedAt: ChronosAPIFixture.now, actuals: window)
        #expect(check.status == expected)
        let observed = availability.filter { $0 == .observed }.count * 20
        #expect(check.metrics.comparedCount == observed && check.metrics.expectedCount == 80)
        #expect((check.metrics.meanAbsoluteError == nil) == (observed == 0))
        #expect(check.metrics.weightedAbsolutePercentageError == nil)
        #expect(check.missingPairs(.unpublished) == availability.filter { $0 == .unpublished }.count * 20)
        #expect(check.missingPairs(.ambiguousTime) == availability.filter { $0 == .ambiguousTime }.count * 20)
        #expect(check.missingPairs(.unverified) == availability.filter { $0 == .unverified }.count * 20)
    }

    @Test("J3 Checks reject wrong origins and zone sets before calculating metrics")
    func correspondence() async throws {
        let record = try await ChronosJournalFixture.record(step: 1)
        for wrongOrigin in [true, false] {
            let window = try APIWindow(
                start: record.input.snapshot.origin.addingTimeInterval(wrongOrigin ? -1800 : 0), count: 4,
                series: record.areas.prefix(wrongOrigin ? 20 : 19).map {
                    CountSeries(areaID: $0.areaID, values: [0, 0, 0, 0])
                }, availability: Array(repeating: .observed, count: 4), datasets: [], fetchedAt: ChronosAPIFixture.now,
                catalog: ChronosAPIFixture.coverage().evidence, metadata: [], responses: [])
            #expect(throws: ChronosDemoFailure.invalidData) {
                try ChronosActualCheck(id: UUID(), record: record, checkedAt: ChronosAPIFixture.now, actuals: window)
            }
        }
    }

    @Test("J1 Capture rejects projected previews and foreign area identities")
    func nativeOnly() async throws {
        let native = try await ChronosJournalFixture.batch()
        for preview in [true, false] {
            let batch = ChronosForecastBatch(
                scenarioID: native.scenarioID, originStep: native.originStep, mode: preview ? .precomputed : .live,
                areas: try native.areas.map {
                    if preview {
                        return try ChronosAreaForecast(
                            areaID: $0.areaID, lower: $0.lower, median: $0.median, upper: $0.upper)
                    }
                    return ChronosAreaForecast(areaID: $0.areaID, raw: try #require($0.raw))
                }, processingSeconds: native.processingSeconds, identity: native.identity,
                capturedInput: native.capturedInput)
            #expect(throws: ChronosDemoFailure.invalidData) {
                try ChronosForecastRecord(
                    id: UUID(), savedAt: ChronosAPIFixture.now, batch: batch, device: ChronosJournalFixture.device)
            }
        }
    }
}
