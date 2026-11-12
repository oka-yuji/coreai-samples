import ChronosFeature
import Foundation
import Testing

@testable import ChronosDemoRuntime

@Suite(.timeLimit(.minutes(1)))
struct ChronosVerifiedRangeTests {
    @Test("The shipped reference has exactly twenty zones and all January half-hours")
    func resource() throws {
        let profile = try NYCVerifiedResource.bundled()
        let source = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appending(
                path: "Sources/ChronosDemoRuntime/Resources/NYCVerified/january-2025.json")
        let original = try NYCVerifiedResource.read(Data(contentsOf: source))
        #expect(profile.policy == original.policy)
        #expect(profile.references == original.references)
        #expect(profile.policy == .january2025)
        #expect(profile.references[0].count == 20)
        #expect(profile.references[0].reduce(0) { $0 + $1.values.count } == 29_760)
        let range = try #require(profile.policy.ranges.first)
        #expect(range.start == (try ChronosAPIFixture.date("2025-01-01T05:00:00Z")))
        #expect(range.end == (try ChronosAPIFixture.date("2025-02-01T05:00:00Z")))
        #expect(try profile.count(at: range.start, areaID: "230", dataset: range.dataset) == 0)
    }

    @Test("Altered bytes and a forged reference identity are rejected")
    func forgedResource() throws {
        let root = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let data = try Data(
            contentsOf: root.appending(path: "Sources/ChronosDemoRuntime/Resources/NYCVerified/january-2025.json"))
        var object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        object["revision"] = "2026-10-02T16:57:47.000Z"
        let forged = try JSONSerialization.data(withJSONObject: object)
        for bytes in [data.dropLast(), data + Data([32]), forged, Data("{}".utf8)] {
            #expect(throws: ChronosDemoFailure.invalidData) { try NYCVerifiedResource.read(Data(bytes)) }
        }
    }

    @Test("Verification profiles reject invalid shapes counts zones and interval boundaries")
    func invalidProfile() throws {
        let profile = try NYCVerifiedResource.bundled()
        let range = try #require(profile.policy.ranges.first)
        let row = try #require(profile.references[0].first)
        var cases = [Array(profile.references[0].dropLast()), profile.references[0] + [row]]
        let invalidValues: [Float?] = [nil, -.infinity, .nan, -1, 0.5, 16_777_218]
        for value in invalidValues {
            var series = profile.references[0]
            var values = row.values
            values[0] = value
            series[0] = CountSeries(areaID: row.areaID, values: values)
            cases.append(series)
        }
        for series in cases {
            #expect(throws: ChronosDemoFailure.invalidData) {
                try NYCVerificationProfile(policy: profile.policy, references: [series])
            }
        }
        let invalid = APIVerifiedRange(
            dataset: range.dataset, start: range.start.addingTimeInterval(1), end: range.end,
            areaIDs: range.areaIDs, referenceSHA256: range.referenceSHA256)
        #expect(throws: ChronosDemoFailure.invalidData) {
            try NYCVerificationProfile(
                policy: APIVerificationPolicy(version: "invalid", ranges: [invalid]), references: profile.references)
        }
    }

    @Test("Sixteen actuals at January end retain the unverified suffix and stop advancement")
    func januaryActuals() async throws {
        let fake = try VerifiedResponseFake()
        let client = fake.client()
        let start = try ChronosAPIFixture.date("2025-02-01T03:00:00Z")
        let history = try await client.window(
            start: start.addingTimeInterval(-1008 * 1800), count: 1008, areaIDs: ["230"])
        let original = try ForecastInputSnapshot(history: history, actuals: nil, step: 0)
        let actuals = try await client.window(start: start, count: 16, areaIDs: ["230"])
        #expect(
            actuals.availability == Array(repeating: .observed, count: 4) + Array(repeating: .unverified, count: 12))
        #expect(actuals.availablePrefix == 4)
        #expect(actuals.series[0].values.suffix(12).allSatisfy { $0 == nil })
        #expect(actuals.responses.count == 1 && actuals.metadata.count == 2)
        #expect(actuals.responses[0].end == (try ChronosAPIFixture.date("2025-02-01T05:00:00Z")))
        #expect(actuals.responses[0].sha256 == NYCAPIContract.hash(actuals.responses[0].data))
        #expect(actuals.responses[0].sha256 != actuals.verification.ranges[0].referenceSHA256)
        let advanced = try ForecastInputSnapshot(history: history, actuals: actuals, step: 4)
        _ = try await client.validate(advanced)
        #expect(advanced.origin == (try ChronosAPIFixture.date("2025-02-01T05:00:00Z")))
        #expect(original.actuals == nil && original.series == history.series)
        #expect(throws: ChronosDemoFailure.unverifiedHistory) {
            try ForecastInputSnapshot(history: history, actuals: actuals, step: 5)
        }
    }

    @Test("Unverified and unlisted spans never make aggregate requests")
    func missingBoundaries() async throws {
        let fake = try VerifiedResponseFake()
        let client = fake.client()
        let window = try await client.window(
            start: ChronosAPIFixture.date("2026-01-01T04:00:00Z"), count: 4, areaIDs: ["230"])
        #expect(window.availability == [.unverified, .unverified, .unpublished, .unpublished])
        #expect(window.series[0].values == [nil, nil, nil, nil])
        #expect(window.responses.isEmpty && window.metadata.isEmpty && window.availablePrefix == 0)
        #expect(await fake.aggregateCount == 0)
        await #expect(throws: ChronosDemoFailure.unverifiedHistory) {
            try await client.window(
                start: ChronosAPIFixture.date("2025-01-12T05:00:00Z"), count: 1008, areaIDs: ["230"])
        }
        await #expect(throws: ChronosDemoFailure.unpublishedHistory) {
            try await client.window(
                start: ChronosAPIFixture.date("2024-12-31T05:00:00Z"), count: 1008, areaIDs: ["230"])
        }
        #expect(await fake.aggregateCount == 0)
    }

    @Test("A verified suffix does not make an earlier unlisted interval available")
    func verifiedSuffix() async throws {
        let fake = try VerifiedResponseFake()
        let window = try await fake.client().window(
            start: ChronosAPIFixture.date("2025-01-01T04:00:00Z"), count: 4, areaIDs: ["230"])
        #expect(window.availability == [.unpublished, .unpublished, .observed, .observed])
        #expect(window.availablePrefix == 0)
        #expect(window.responses[0].start == (try ChronosAPIFixture.date("2025-01-01T05:00:00Z")))
    }

    @Test("Metadata changes before or after aggregation reject the window", arguments: [1, 2])
    func revisionChange(_ changedCall: Int) async throws {
        let fake = try VerifiedResponseFake(changedMetadataCall: changedCall)
        await #expect(throws: ChronosDemoFailure.invalidData) {
            try await fake.client().window(
                start: ChronosAPIFixture.date("2025-01-01T05:00:00Z"), count: 1, areaIDs: ["230"])
        }
        #expect(await fake.aggregateCount == (changedCall == 1 ? 0 : 1))
    }

    @Test("A nonexistent spring timestamp is rejected rather than reassigned")
    func invalidSpringRow() async throws {
        let fake = try ChronosAPITransportFake(rows: ChronosAPIFixture.rows(day: "2025-03-09T00:00:00.000", hour: "2"))
        let client = try ChronosAPIFixture.client(fake)
        await #expect(throws: ChronosDemoFailure.invalidData) {
            try await client.window(start: ChronosAPIFixture.date("2025-03-09T06:00:00Z"), count: 6, areaIDs: ["230"])
        }
    }

    @Test(
        "Old policy altered counts and mixed revision caches are revalidated before reuse",
        arguments: ["legacy", "policy", "counts", "metadata"])
    func cachePolicy(_ alteration: String) async throws {
        let root = try ChronosAPIFixture.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let fake = try VerifiedResponseFake()
        let client = fake.client(cache: root)
        let start = try ChronosAPIFixture.date("2025-01-24T21:30:00Z")
        let window = try await client.window(start: start, count: 1, areaIDs: ["230"])
        let file = try #require(
            FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil).first {
                $0.pathExtension == "json"
            })
        var envelope = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
        let encodedPayload = try #require(envelope["payload"] as? String)
        let payload = try #require(Data(base64Encoded: encodedPayload))
        var object = try #require(JSONSerialization.jsonObject(with: payload) as? [String: Any])
        switch alteration {
        case "legacy": object.removeValue(forKey: "verification")
        case "policy": object["verification"] = ["version": "old-broad-year", "ranges": []]
        case "counts":
            var responses = try #require(object["responses"] as? [[String: Any]])
            let empty = Data("[]".utf8)
            responses[0]["data"] = empty.base64EncodedString()
            responses[0]["sha256"] = NYCAPIContract.hash(empty)
            object["responses"] = responses
            object["series"] = [["areaID": "230", "values": [0]]]
        default:
            var records = try #require(object["metadata"] as? [[String: Any]])
            let encoded = try #require(records[1]["data"] as? String)
            let bytes = try #require(Data(base64Encoded: encoded))
            var metadata = try #require(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
            metadata["rowsUpdatedAt"] = 1_790_873_868
            let changed = try JSONSerialization.data(withJSONObject: metadata)
            records[1]["data"] = changed.base64EncodedString()
            records[1]["sha256"] = NYCAPIContract.hash(changed)
            object["metadata"] = records
        }
        let changed = try JSONSerialization.data(withJSONObject: object)
        envelope["payload"] = changed.base64EncodedString()
        envelope["sha256"] = NYCAPIContract.hash(changed)
        try JSONSerialization.data(withJSONObject: envelope).write(to: file)
        #expect(throws: ChronosDemoFailure.invalidData) {
            try NYCWindowCache(directory: root, capacity: 32).read(
                file.deletingPathExtension().lastPathComponent, profile: NYCVerifiedResource.bundled())
        }
        let repaired = try await client.window(start: start, count: 1, areaIDs: ["230"])
        #expect(repaired == window)
        #expect(await fake.aggregateCount == 2)
    }
}

actor VerifiedResponseFake {
    let profile: NYCVerificationProfile
    let catalog: Data
    let metadata: Data
    let changedMetadataCall: Int?
    var metadataCount = 0
    var aggregateCount = 0

    init(changedMetadataCall: Int? = nil) throws {
        profile = try NYCVerifiedResource.bundled()
        catalog = try ChronosAPIFixture.catalog()
        metadata = try ChronosAPIFixture.metadata()
        self.changedMetadataCall = changedMetadataCall
    }

    nonisolated func client(cache: URL? = nil) -> NYCAPIClient {
        NYCAPIClient(
            cacheDirectory: cache, transport: { try await self.receive($0) }, now: { ChronosAPIFixture.now },
            retryDelay: {})
    }

    func receive(_ request: URLRequest) throws -> NYCAPIResponse {
        let url = try #require(request.url)
        if url.host == "api.us.socrata.com" { return NYCAPIResponse(data: catalog, status: 200) }
        if url.path.hasPrefix("/api/views/") {
            metadataCount += 1
            if metadataCount == changedMetadataCall {
                var changed = try #require(JSONSerialization.jsonObject(with: metadata) as? [String: Any])
                changed["rowsUpdatedAt"] = 1_790_873_868
                return NYCAPIResponse(data: try JSONSerialization.data(withJSONObject: changed), status: 200)
            }
            return NYCAPIResponse(data: metadata, status: 200)
        }
        aggregateCount += 1
        let query = try #require(
            URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.first { $0.name == "$where" }?.value)
        let bounds = query.split(separator: "'")
        try #require(bounds.count == 5)
        let areas = try #require(query.split(separator: "(").last?.split(separator: ")").first).split(separator: ",")
            .map(String.init)
        let range = try #require(profile.policy.ranges.first)
        var rows: [[String: Any]] = []
        for index in 0..<1488 {
            let date = range.start.addingTimeInterval(Double(index * 1800))
            let key = NYCTime.key(date)
            guard key >= bounds[1], key < bounds[3] else { continue }
            for row in profile.references[0] where areas.contains(row.areaID) {
                let count = try #require(row.values[index])
                guard count > 0 else { continue }
                rows.append([
                    "day": String(key.prefix(10)) + "T00:00:00.000",
                    "hour": String(NYCTime.calendar.component(.hour, from: date)),
                    "second_half": NYCTime.calendar.component(.minute, from: date) == 30,
                    "zone": row.areaID, "trips": String(Int(count)),
                ])
            }
        }
        return NYCAPIResponse(data: try JSONSerialization.data(withJSONObject: rows, options: .sortedKeys), status: 200)
    }
}
