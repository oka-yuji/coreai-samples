import ChronosFeature
import Foundation
import Testing

@testable import ChronosDemoRuntime

@Suite(.timeLimit(.minutes(1)))
struct ChronosAPIValidationTests {
    @Test(
        "A1 Malformed revision timestamps cannot authorize an eligible dataset",
        arguments: [
            "2026-10-01T16:57:47.000Zgarbage", "2026-02-30T16:57:47.000Z",
            "2026-10-01T24:00:00.000Z", "2026-10-01T16:57:47.000Z\n",
        ])
    func malformedRevision(_ revision: String) async throws {
        let dataset = APIDataset(year: 2025, id: "irx2-vgm4", revision: revision)
        let fake = try ChronosAPITransportFake(catalog: ChronosAPIFixture.catalog([ChronosAPIFixture.item(dataset)]))
        await #expect(throws: ChronosDemoFailure.invalidData) {
            try await ChronosAPIFixture.client(fake).coverage()
        }
    }

    @Test("A1 Canonical UTC revision timestamps retain their exact epoch")
    func canonicalRevision() {
        for text in ["2026-10-01T16:57:47.000Z", "2026-10-01T16:57:47Z"] {
            #expect(NYCCatalogValidation.timestamp(text)?.timeIntervalSince1970 == 1_790_873_867)
        }
    }

    @Test("A1 A late publication date cannot authorize rows last updated before the annual period closed")
    func incompleteAnnualRevision() throws {
        let dataset = APIDataset(year: 2025, id: "irx2-vgm4", revision: "2025-06-01T00:00:00.000Z")
        var metadata = try #require(
            JSONSerialization.jsonObject(with: ChronosAPIFixture.metadata(dataset)) as? [String: Any])
        metadata["rowsUpdatedAt"] = try ChronosAPIFixture.date("2025-06-01T00:00:00Z").timeIntervalSince1970
        let evidence = try ChronosAPIFixture.evidence(
            JSONSerialization.data(withJSONObject: metadata), url: NYCAPIContract.metadataURL(dataset))
        #expect(throws: ChronosDemoFailure.invalidData) {
            try NYCCatalogValidation.metadata(evidence, dataset: dataset)
        }
    }

    @Test("A1 An as-needed update label alone does not establish a historical annual publication")
    func unprovenHistoricalMetadata() throws {
        var metadata = try #require(JSONSerialization.jsonObject(with: ChronosAPIFixture.metadata()) as? [String: Any])
        metadata["metadata"] = [
            "custom_fields": [
                "Update": ["Update Frequency": "As needed"],
                "Dataset Information": ["Agency": "Taxi and Limousine Commission (TLC)"],
            ]
        ]
        let evidence = try ChronosAPIFixture.evidence(
            JSONSerialization.data(withJSONObject: metadata),
            url: NYCAPIContract.metadataURL(ChronosAPIFixture.dataset()))
        #expect(throws: ChronosDemoFailure.invalidData) {
            try NYCCatalogValidation.metadata(evidence, dataset: ChronosAPIFixture.dataset())
        }
    }

    @Test("A1 Metadata must prove the same historical annual publication and dataset revision")
    func metadataIdentity() throws {
        let original = try #require(JSONSerialization.jsonObject(with: ChronosAPIFixture.metadata()) as? [String: Any])
        let changes: [(String, Any)] = [
            ("id", "aaaa-bbbb"), ("name", "2024 Yellow Taxi Trip Data"), ("owner", ["id": "other"]),
            ("provenance", "community"), ("publicationStage", "unpublished"),
            ("publicationDate", 0), ("publicationDate", ChronosAPIFixture.now.timeIntervalSince1970 + 1),
            ("rowsUpdatedAt", 1_790_873_868),
            ("metadata", ["custom_fields": ["Update": ["Update Frequency": "Monthly"]]]),
        ]
        for (key, value) in changes {
            var object = original
            object[key] = value
            let evidence = try ChronosAPIFixture.evidence(
                JSONSerialization.data(withJSONObject: object),
                url: NYCAPIContract.metadataURL(ChronosAPIFixture.dataset()))
            #expect(throws: ChronosDemoFailure.invalidData) {
                try NYCCatalogValidation.metadata(evidence, dataset: ChronosAPIFixture.dataset())
            }
        }
    }

    @Test("A1 Only trusted closed annual datasets are selectable and gaps reject history")
    func catalogTrust() throws {
        var untrusted = ChronosAPIFixture.item()
        untrusted["owner"] = ["id": "other"]
        let open = ChronosAPIFixture.item(ChronosAPIFixture.dataset(2026))
        let data = try ChronosAPIFixture.catalog([ChronosAPIFixture.item(), untrusted, open])
        let coverage = try NYCCatalogValidation.read(
            ChronosAPIFixture.evidence(data, url: NYCAPIContract.catalogURL()), now: ChronosAPIFixture.now)
        #expect(coverage.datasets == [ChronosAPIFixture.dataset()])
        let gap = APICoverage(
            datasets: [ChronosAPIFixture.dataset(2023), ChronosAPIFixture.dataset()], evidence: coverage.evidence)
        #expect(gap.selectableRanges.count == 2)
        #expect(!gap.allowsHistory(endingAt: try ChronosAPIFixture.date("2025-01-05T22:00:00Z")))
        #expect(gap.allowsHistory(endingAt: try ChronosAPIFixture.date("2025-01-24T22:00:00Z")))
    }

    @Test("A1 Eligible malformed and duplicate years fail instead of becoming unpublished")
    func malformedCatalog() throws {
        var malformed = ChronosAPIFixture.item()
        malformed["resource"] = ["id": "irx2-vgm4", "name": "2025 Yellow Taxi Trip Data", "provenance": "official"]
        for items in [[malformed], [ChronosAPIFixture.item(), ChronosAPIFixture.item()]] {
            let evidence = try ChronosAPIFixture.evidence(
                ChronosAPIFixture.catalog(items), url: NYCAPIContract.catalogURL())
            #expect(throws: ChronosDemoFailure.invalidData) {
                try NYCCatalogValidation.read(evidence, now: ChronosAPIFixture.now)
            }
        }
    }

    @Test("A1 Metadata identity and floating timestamp types are validated before zero filling")
    func metadataTypes() async throws {
        let fake = try ChronosAPITransportFake(metadata: ["irx2-vgm4": ChronosAPIFixture.metadata(type: "text")])
        let client = ChronosAPIFixture.client(fake)
        let date = try ChronosAPIFixture.date("2025-01-03T22:00:00Z")
        await #expect(throws: ChronosDemoFailure.invalidData) {
            try await client.window(start: date, count: 1, areaIDs: ["230"])
        }
        #expect(await fake.requests.count == 2)
    }

    @Test("A2 Entire published zero histories are valid and unlisted years stay missing")
    func zeros() async throws {
        let fake = try ChronosAPITransportFake()
        let client = ChronosAPIFixture.client(fake)
        let start = try ChronosAPIFixture.date("2025-01-03T22:00:00Z")
        let history = try await client.window(start: start, count: 1008, areaIDs: ["230"])
        #expect(history.series[0].values == Array(repeating: 0, count: 1008))
        #expect(history.responses.count == 3 && history.availability.allSatisfy { $0 == .observed })
        let snapshot = try ForecastInputSnapshot(history: history, actuals: nil, step: 0)
        #expect(snapshot.actuals == nil && snapshot.origin == start.addingTimeInterval(1008 * 1800))
        let future = try await client.window(
            start: ChronosAPIFixture.date("2026-01-01T05:00:00Z"), count: 16, areaIDs: ["230"])
        #expect(future.availability.allSatisfy { $0 == .unpublished })
        #expect(future.series[0].values.allSatisfy { $0 == nil })
        let calls = await fake.requests.count
        await #expect(throws: ChronosDemoFailure.unpublishedHistory) {
            try await client.window(
                start: ChronosAPIFixture.date("2024-12-20T05:00:00Z"), count: 1008, areaIDs: ["230"])
        }
        #expect(await fake.requests.count == calls)
    }

    @Test("A1 Provider day strings and integer counts must be complete canonical values")
    func rows() async throws {
        let cases = try [
            ChronosAPIFixture.rows(day: "2025-01-03garbage"),
            ChronosAPIFixture.rows(day: "2025-01-03T01:00:00.000"),
            ChronosAPIFixture.rows(day: "2025-02-30T00:00:00.000"),
            ChronosAPIFixture.rows(day: "2024-01-03T00:00:00.000"),
            ChronosAPIFixture.rows(hour: "24"), ChronosAPIFixture.rows(zone: "0230"),
            ChronosAPIFixture.rows(trips: "NaN"), ChronosAPIFixture.rows(trips: "-1"),
            ChronosAPIFixture.rows(trips: "1.00000000000000001"), ChronosAPIFixture.rows(trips: "16777217"),
            ChronosAPIFixture.rows(copies: 2), ChronosAPIFixture.rows(copies: 50_000),
        ]
        for data in cases {
            let client = ChronosAPIFixture.client(try ChronosAPITransportFake(rows: data))
            await #expect(throws: ChronosDemoFailure.invalidData) {
                try await client.window(
                    start: ChronosAPIFixture.date("2025-01-03T22:00:00Z"), count: 1, areaIDs: ["230"])
            }
        }
    }

    @Test("A2 Both autumn folds are missing while spring UTC slots stay consecutive")
    func daylightSaving() async throws {
        let client = ChronosAPIFixture.client(try ChronosAPITransportFake())
        let autumn = try await client.window(
            start: ChronosAPIFixture.date("2025-11-02T04:00:00Z"), count: 8, areaIDs: ["230"])
        #expect(
            autumn.availability == [
                .observed, .observed, .ambiguousTime, .ambiguousTime, .ambiguousTime, .ambiguousTime, .observed,
                .observed,
            ])
        #expect(autumn.series[0].values == [0, 0, nil, nil, nil, nil, 0, 0])
        let spring = try await client.window(
            start: ChronosAPIFixture.date("2025-03-09T06:00:00Z"), count: 6, areaIDs: ["230"])
        #expect(spring.availability.allSatisfy { $0 == .observed })
        #expect(NYCTime.key(spring.start.addingTimeInterval(3600)) == "2025-03-09T03:00:00")
    }

    @Test("A1 Queries split at New York year boundaries and at seven days")
    func splitYears() async throws {
        let old = ChronosAPIFixture.dataset(2024)
        let fake = try ChronosAPITransportFake(
            catalog: ChronosAPIFixture.catalog([ChronosAPIFixture.item(old), ChronosAPIFixture.item()]),
            metadata: [old.id: ChronosAPIFixture.metadata(old), "irx2-vgm4": ChronosAPIFixture.metadata()])
        let value = try await ChronosAPIFixture.client(fake).window(
            start: ChronosAPIFixture.date("2024-12-25T05:00:00Z"), count: 1008, areaIDs: ["230"])
        #expect(value.responses.count == 3 && value.metadata.count == 2)
        #expect(
            value.responses.allSatisfy {
                ($0.end ?? .distantFuture).timeIntervalSince($0.start ?? .distantPast) <= 7 * 86400
            })
        #expect(value.responses[0].end == (try ChronosAPIFixture.date("2025-01-01T05:00:00Z")))
    }

    @Test("A1 Failure kinds stay distinct and transient retry occurs only once")
    func failures() async throws {
        for (status, failure) in [(429, ChronosDemoFailure.rateLimited), (302, .redirectDenied), (404, .network)] {
            let fake = try ChronosAPITransportFake()
            await fake.setStatus(status)
            await #expect(throws: failure) { try await ChronosAPIFixture.client(fake).coverage() }
            #expect(await fake.requests.count == 1)
        }
        let retry = try ChronosAPITransportFake()
        await retry.setStatuses([503, 503])
        await #expect(throws: ChronosDemoFailure.network) { try await ChronosAPIFixture.client(retry).coverage() }
        #expect(await retry.requests.count == 2)
        let large = try ChronosAPITransportFake(catalog: Data(repeating: 0, count: NYCAPIContract.byteLimit + 1))
        await #expect(throws: ChronosDemoFailure.responseTooLarge) {
            try await ChronosAPIFixture.client(large).coverage()
        }
        for text in [
            "http://data.cityofnewyork.us/resource/irx2-vgm4.json",
            "https://data.cityofnewyork.us:444/resource/irx2-vgm4.json", "https://other.test/resource/irx2-vgm4.json",
            "https://data.cityofnewyork.us/elsewhere",
        ] {
            #expect(!NYCAPIContract.validURL(try #require(URL(string: text))))
        }
    }
}
