import ChronosCore
import ChronosFeature
import CryptoKit
import Foundation

public actor ChronosRepository {
    private let directory: URL
    private var loaded: ChronosCatalog?
    private var frames: [CachedFrame] = []

    public init(directory: URL) { self.directory = directory }

    public static func bundled() throws -> ChronosRepository {
        ChronosRepository(directory: try bundledDirectory())
    }

    static func bundledDirectory() throws -> URL {
        guard let directory = Bundle.module.url(forResource: "NYCData", withExtension: nil) else {
            throw ChronosDemoFailure.invalidData
        }
        return directory
    }

    public func load() throws -> ChronosCatalog {
        if let loaded { return loaded }
        let verified = try validateFiles()
        let manifest: SourceManifest = try read("manifest.json", from: verified)
        let city: City = try read("city.json", from: verified)
        let cache: CachedCollection = try read("forecasts.json", from: verified)
        guard manifest.synthetic == city.synthetic, city.areas.count == 20, !manifest.synthetic,
            Set(city.areas.map(\.id)).count == city.areas.count,
            manifest.time.historySteps == 1008, manifest.time.forecastSteps == 4,
            manifest.time.simulationFutureSteps == 16, manifest.time.intervalSeconds == 1800,
            let origin = ISO8601DateFormatter().date(from: manifest.time.forecastOrigin),
            cache.modelRevision == ChronosContract.revision, cache.assetSHA256 == Self.graphSHA256
        else { throw ChronosDemoFailure.invalidData }
        if !manifest.synthetic {
            let metadata = try bytes("manifest.json", from: verified)
            let metadataHash = SHA256.hash(data: metadata).map { String(format: "%02x", $0) }.joined()
            let data = try bytes("city.json", from: verified)
            let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard metadataHash == cache.sourceHashes["manifest.json"],
                hash == manifest.citySha256, cache.sourceHashes["city.json"] == hash,
                manifest.time.timezone == "America/New_York", manifest.source != nil,
                city.areas.allSatisfy({ area in
                    guard let center = area.center, center.isValid, let polygons = area.polygons else { return false }
                    return !polygons.isEmpty && polygons.allSatisfy(\.isValid)
                        && polygons.contains { $0.contains(center) }
                })
            else { throw ChronosDemoFailure.invalidData }
        }
        let ids = Set(city.areas.map(\.id))
        var scenarios: [ChronosScenario] = []
        for item in manifest.scenarios {
            let historyData = try bytes(item.inputFile, from: verified)
            let hash = SHA256.hash(data: historyData).map { String(format: "%02x", $0) }.joined()
            guard hash == item.inputSha256, cache.sourceHashes[item.inputFile] == hash else {
                throw ChronosDemoFailure.invalidData
            }
            let input = try decoder().decode(SeriesFile.self, from: historyData)
            let truthData = try bytes(item.truthFile, from: verified)
            let truthHash = SHA256.hash(data: truthData).map { String(format: "%02x", $0) }.joined()
            guard truthHash == item.truthSha256, cache.sourceHashes[item.truthFile] == truthHash else {
                throw ChronosDemoFailure.invalidData
            }
            let truth = try decoder().decode(SeriesFile.self, from: truthData)
            guard input.synthetic == manifest.synthetic, truth.synthetic == manifest.synthetic,
                input.scenarioId == item.id, truth.scenarioId == item.id,
                ids.contains(item.focusArea)
            else { throw ChronosDemoFailure.invalidData }
            let scenarioOrigin = item.forecastOrigin.flatMap { ISO8601DateFormatter().date(from: $0) }
            guard manifest.synthetic || scenarioOrigin != nil else { throw ChronosDemoFailure.invalidData }
            let history = try validated(input.series, ids: ids, count: manifest.time.historySteps)
            let future = try validated(truth.series, ids: ids, count: manifest.time.simulationFutureSteps)
            scenarios.append(
                ChronosScenario(
                    id: item.id, name: item.name, focusArea: item.focusArea,
                    history: history, truth: future, origin: scenarioOrigin))
        }
        guard scenarios.count == 3, Set(scenarios.map(\.id)).count == 3 else { throw ChronosDemoFailure.invalidData }
        let scenarioIDs = Set(scenarios.map(\.id))
        var seen: Set<String> = []
        for frame in cache.frames {
            guard scenarioIDs.contains(frame.scenarioID), (0...12).contains(frame.step),
                seen.insert(frame.scenarioID + "-" + String(frame.step)).inserted,
                Set(frame.areas.map(\.areaID)) == ids, frame.areas.count == ids.count
            else {
                throw ChronosDemoFailure.invalidData
            }
            _ = try frame.validatedAreas()
        }
        guard cache.frames.count == 39 else { throw ChronosDemoFailure.invalidData }
        let catalog = ChronosCatalog(
            areas: city.areas, scenarios: scenarios, origin: origin, intervalSeconds: 1800,
            historySteps: manifest.time.historySteps, truthSteps: 16,
            timeZone: "America/New_York", isSynthetic: false,
            sourceURL: manifest.source?.url, dataPeriod: manifest.source?.period ?? "")
        frames = cache.frames
        loaded = catalog
        return catalog
    }

    public func cached(_ request: ChronosForecastRequest) throws -> ChronosForecastBatch {
        try Task.checkCancellation()
        let catalog = try load()
        guard request.mode == .precomputed,
            catalog.request(scenarioID: request.scenarioID, step: request.originStep, mode: .precomputed) == request,
            let frame = frames.first(where: { $0.scenarioID == request.scenarioID && $0.step == request.originStep })
        else { throw ChronosDemoFailure.invalidData }
        return ChronosForecastBatch(
            scenarioID: request.scenarioID, originStep: request.originStep, mode: .precomputed,
            areas: try frame.validatedAreas(), processingSeconds: nil)
    }

    func capture(_ request: ChronosForecastRequest) throws -> ChronosInputEvidence {
        let catalog = try load()
        let capture: ChronosInputCapture
        if let snapshot = request.snapshot {
            capture = ChronosInputCapture(
                source: .api, scenarioID: request.scenarioID, originStep: request.originStep,
                origin: snapshot.origin, series: snapshot.series, preprocessing: snapshot.preprocessing, api: snapshot)
        } else {
            let files = try validateFiles()
            let names = [
                "manifest.json", "city.json", request.scenarioID + ".input.json", request.scenarioID + ".truth.json",
            ]
            capture = ChronosInputCapture(
                source: .fixed, scenarioID: request.scenarioID, originStep: request.originStep,
                origin: catalog.time(at: request.originStep, scenarioID: request.scenarioID),
                series: request.series.map {
                    CountSeries(areaID: $0.areaID, values: $0.values.map { $0.isNaN ? nil : $0 })
                },
                preprocessing: "nyc-fixed-utc30m-v1", fixedFiles: files.filter { names.contains($0.key) })
        }
        guard
            capture.series.map({ ChronosSeriesInput(areaID: $0.areaID, values: $0.values.map { $0 ?? .nan }) })
                == request.series
        else { throw ChronosDemoFailure.invalidData }
        let bytes = try NYCAPIContract.encode(capture)
        let evidence = ChronosInputEvidence(snapshot: capture, bytes: bytes, sha256: NYCAPIContract.hash(bytes))
        if capture.source == .fixed { try Self.validateFixed(capture) }
        return evidence
    }

    static func validateFixed(_ input: ChronosInputCapture) throws {
        let files = input.fixedFiles
        let names = ["manifest.json", "city.json", input.scenarioID + ".input.json", input.scenarioID + ".truth.json"]
        guard Set(files.keys) == Set(names), input.preprocessing == "nyc-fixed-utc30m-v1", input.api == nil,
            files.allSatisfy({ hashes[$0.key] == NYCAPIContract.hash($0.value) })
        else { throw ChronosDemoFailure.invalidData }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let manifestBytes = files["manifest.json"], let cityBytes = files["city.json"],
            let historyBytes = files[input.scenarioID + ".input.json"],
            let truthBytes = files[input.scenarioID + ".truth.json"]
        else { throw ChronosDemoFailure.invalidData }
        let manifest = try decoder.decode(SourceManifest.self, from: manifestBytes)
        let city = try decoder.decode(City.self, from: cityBytes)
        let history = try decoder.decode(SeriesFile.self, from: historyBytes)
        let truth = try decoder.decode(SeriesFile.self, from: truthBytes)
        guard let scenario = manifest.scenarios.first(where: { $0.id == input.scenarioID }),
            let originText = scenario.forecastOrigin, let origin = ISO8601DateFormatter().date(from: originText),
            input.origin == origin.addingTimeInterval(Double(input.originStep * 1800)),
            input.series.map(\.areaID) == city.areas.map(\.id)
        else { throw ChronosDemoFailure.invalidData }
        for series in input.series {
            guard let past = history.series.first(where: { $0.areaId == series.areaID }),
                let future = truth.series.first(where: { $0.areaId == series.areaID }),
                let values = ChronosHistoryWindow.make(
                    history: past.counts, truth: future.counts, revealed: input.originStep, maxLength: 1008),
                series.values == values.map(Optional.some)
            else { throw ChronosDemoFailure.invalidData }
        }
    }

    public static let graphSHA256 = "3094871fc431ae5d86d07c8b487ab53e48e60e9fb56efe803ba18c5bf88889c8"
    static let hashes: [String: String] = [
        "city.json": "de17bc4a6ef36b340aeb709779d3765b846cf9f03680df3e751f32f7f15495f0",
        "forecasts.json": "8de8c40c66a33466e42c493b3c9f93b6ab2748883a18798d7d1c093ca311899b",
        "friday-evening.input.json": "099070a7fed88f9041c9bee110ed80dba0998697a4bc203ea1231899035d4f7e",
        "friday-evening.truth.json": "23389b89a06a94ae5d1d6ab0d615cf769331ee9acf0dfb8780490e7a22b88826",
        "manifest.json": "7c3e0575487885d64c0d01e8639cce824c21d0a74c45300ee00c042a363fed03",
        "saturday-night.input.json": "5a0b638761360987df3d79174a9baaa35b94d8d9e33ef967c547125b97b47f8f",
        "saturday-night.truth.json": "8ba4e4dcaef74abaa1b741f46c4642269a38e0095f9f2feb8f25576dba1fc184",
        "weekday-morning.input.json": "c36832b3d90cbbab2328f3f7b25e53cc3d02edf37da2fbc06354f1516ea13063",
        "weekday-morning.truth.json": "fe30406b3dedef0380ae070741ca3b0b8d9b09c28983d8edf8981eb41027d91f",
    ]

    private func validateFiles() throws -> [String: Data] {
        var verified: [String: Data] = [:]
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        guard Set(names) == Set(Self.hashes.keys) else { throw ChronosDemoFailure.invalidData }
        for (name, expected) in Self.hashes {
            let url = directory.appendingPathComponent(name)
            let metadata = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            guard metadata.isRegularFile == true, metadata.isSymbolicLink != true else {
                throw ChronosDemoFailure.invalidData
            }
            let data = try Data(contentsOf: url)
            let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard hash == expected else { throw ChronosDemoFailure.invalidData }
            verified[name] = data
        }
        return verified
    }

    private func bytes(_ name: String, from verified: [String: Data]) throws -> Data {
        guard let data = verified[name] else { throw ChronosDemoFailure.invalidData }
        return data
    }

    private func read<T: Decodable>(_ name: String, from verified: [String: Data]) throws -> T {
        try decoder().decode(T.self, from: bytes(name, from: verified))
    }

    private func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }

    private func validated(_ series: [SeriesFile.Row], ids: Set<String>, count: Int) throws -> [String: [Float]] {
        guard series.count == ids.count, Set(series.map(\.areaId)) == ids,
            series.allSatisfy({ $0.counts.count == count && $0.counts.allSatisfy { $0.isFinite && $0 >= 0 } })
        else { throw ChronosDemoFailure.invalidData }
        return Dictionary(uniqueKeysWithValues: series.map { ($0.areaId, $0.counts) })
    }
}

private struct City: Decodable {
    let synthetic: Bool
    let areas: [ChronosArea]
}

private struct SourceManifest: Decodable {
    struct Timing: Decodable {
        let forecastOrigin: String
        let intervalSeconds: Int
        let historySteps: Int
        let forecastSteps: Int
        let simulationFutureSteps: Int
        let timezone: String?
    }

    struct Scenario: Decodable {
        let id: String
        let name: ChronosDisplayName
        let focusArea: String
        let inputFile: String
        let truthFile: String
        let inputSha256: String
        let truthSha256: String
        let forecastOrigin: String?
    }

    struct Source: Decodable {
        let url: String
        let period: String
    }

    let synthetic: Bool
    let time: Timing
    let scenarios: [Scenario]
    let citySha256: String?
    let source: Source?
}

private struct SeriesFile: Decodable {
    struct Row: Decodable {
        let areaId: String
        let counts: [Float]
    }

    let synthetic: Bool
    let scenarioId: String
    let series: [Row]
}

private struct CachedCollection: Decodable {
    let modelRevision: String
    let assetSHA256: String
    let sourceHashes: [String: String]
    let frames: [CachedFrame]
}

private struct CachedFrame: Decodable {
    struct Row: Decodable {
        let areaID: String
        let lower: [Float]
        let median: [Float]
        let upper: [Float]
    }

    let scenarioID: String
    let step: Int
    let areas: [Row]

    func validatedAreas() throws -> [ChronosAreaForecast] {
        try areas.map {
            try ChronosAreaForecast(areaID: $0.areaID, lower: $0.lower, median: $0.median, upper: $0.upper)
        }
    }
}
