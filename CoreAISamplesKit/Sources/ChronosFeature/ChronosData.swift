import ChronosCore
import Foundation

public struct ChronosDisplayName: Codable, Equatable, Sendable {
    public let ja: String
    public let en: String

    public init(ja: String, en: String) {
        self.ja = ja
        self.en = en
    }
}

public struct ChronosArea: Codable, Equatable, Identifiable, Sendable {
    public struct Grid: Codable, Equatable, Sendable {
        public let x: Int
        public let y: Int

        public init(x: Int, y: Int) {
            self.x = x
            self.y = y
        }
    }

    public let id: String
    public let name: ChronosDisplayName
    public let district: String
    public let grid: Grid
    public let center: ChronosGeoPoint?
    public let polygons: [ChronosZonePolygon]?

    public init(
        id: String, name: ChronosDisplayName, district: String, grid: Grid, center: ChronosGeoPoint? = nil,
        polygons: [ChronosZonePolygon]? = nil
    ) {
        self.id = id
        self.name = name
        self.district = district
        self.grid = grid
        self.center = center
        self.polygons = polygons
    }
}

public struct ChronosScenario: Equatable, Identifiable, Sendable {
    public let id: String
    public let name: ChronosDisplayName
    public let focusArea: String
    public let history: [String: [Float]]
    public let truth: [String: [Float]]
    public let origin: Date?
    public let apiHistory: APIWindow?
    public let apiActuals: APIWindow?

    public init(
        id: String, name: ChronosDisplayName, focusArea: String, history: [String: [Float]], truth: [String: [Float]],
        origin: Date? = nil, apiHistory: APIWindow? = nil, apiActuals: APIWindow? = nil
    ) {
        self.id = id
        self.name = name
        self.focusArea = focusArea
        self.history = history
        self.truth = truth
        self.origin = origin
        self.apiHistory = apiHistory
        self.apiActuals = apiActuals
    }
}

public enum ChronosForecastMode: String, Codable, CaseIterable, Hashable, Sendable {
    case live
    case precomputed
}

public enum ChronosDemoFailure: String, Error, Equatable, Sendable {
    case invalidData
    case unavailable
    case busy
    case predictionFailed
    case invalidDate
    case unpublishedHistory
    case unverifiedHistory
    case noData
    case network
    case rateLimited
    case responseTooLarge
    case redirectDenied
    case persistence
}

public struct ChronosSeriesInput: Equatable, Sendable {
    public let areaID: String
    public let values: [Float]

    public init(areaID: String, values: [Float]) {
        self.areaID = areaID
        self.values = values
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.areaID == rhs.areaID && lhs.values.count == rhs.values.count
            && zip(lhs.values, rhs.values).allSatisfy { $0 == $1 || ($0.isNaN && $1.isNaN) }
    }
}

public struct ChronosForecastRequest: Equatable, Sendable {
    public let scenarioID: String
    public let originStep: Int
    public let mode: ChronosForecastMode
    public let series: [ChronosSeriesInput]
    public let snapshot: ForecastInputSnapshot?

    public init(
        scenarioID: String, originStep: Int, mode: ChronosForecastMode, series: [ChronosSeriesInput],
        snapshot: ForecastInputSnapshot? = nil
    ) {
        self.scenarioID = scenarioID
        self.originStep = originStep
        self.mode = mode
        self.series = series
        self.snapshot = snapshot
    }
}

public struct ChronosAreaForecast: Equatable, Encodable, Sendable {
    public let areaID: String
    public let raw: ChronosPrediction?
    private let projection: [[Float]]
    public var lower: [Float] { projection[0].map { max(0, $0) } }
    public var median: [Float] { projection[1].map { max(0, $0) } }
    public var upper: [Float] { projection[2].map { max(0, $0) } }

    public init(areaID: String, lower: [Float], median: [Float], upper: [Float]) throws {
        guard [lower, median, upper].allSatisfy({ $0.count == 4 && $0.allSatisfy(\.isFinite) }) else {
            throw ChronosDemoFailure.predictionFailed
        }
        self.areaID = areaID
        raw = nil
        projection = [lower, median, upper]
    }
}

extension ChronosAreaForecast {
    public init(areaID: String, raw: ChronosPrediction) {
        self.areaID = areaID
        self.raw = raw
        projection = [2, 10, 18].map { Array(raw.quantiles[$0].prefix(4)) }
    }
}

public struct ChronosForecastBatch: Equatable, Encodable, Sendable {
    public let scenarioID: String
    public let originStep: Int
    public let mode: ChronosForecastMode
    public let areas: [ChronosAreaForecast]
    public let processingSeconds: Double?
    public let identity: ChronosBundleIdentity?
    public let inputSnapshot: ForecastInputSnapshot?
    public let inputSHA256: String?

    public init(
        scenarioID: String, originStep: Int, mode: ChronosForecastMode, areas: [ChronosAreaForecast],
        processingSeconds: Double?, identity: ChronosBundleIdentity? = nil,
        inputSnapshot: ForecastInputSnapshot? = nil, inputSHA256: String? = nil
    ) {
        self.scenarioID = scenarioID
        self.originStep = originStep
        self.mode = mode
        self.areas = areas
        self.processingSeconds = processingSeconds
        self.identity = identity
        self.inputSnapshot = inputSnapshot
        self.inputSHA256 = inputSHA256
    }

    public func area(_ id: String) -> ChronosAreaForecast? { areas.first { $0.areaID == id } }
}

public struct ChronosCatalog: Equatable, Sendable {
    public let areas: [ChronosArea]
    public let scenarios: [ChronosScenario]
    public let origin: Date
    public let intervalSeconds: Int
    public let historySteps: Int
    public let truthSteps: Int
    public let horizon: Int
    public let timeZone: String
    public let isSynthetic: Bool
    public let sourceURL: String?
    public let dataPeriod: String

    public init(
        areas: [ChronosArea], scenarios: [ChronosScenario], origin: Date, intervalSeconds: Int,
        historySteps: Int, truthSteps: Int, horizon: Int = 4,
        timeZone: String = "America/New_York", isSynthetic: Bool = false, sourceURL: String? = nil,
        dataPeriod: String = ""
    ) {
        self.areas = areas
        self.scenarios = scenarios
        self.origin = origin
        self.intervalSeconds = intervalSeconds
        self.historySteps = historySteps
        self.truthSteps = truthSteps
        self.horizon = horizon
        self.timeZone = timeZone
        self.isSynthetic = isSynthetic
        self.sourceURL = sourceURL
        self.dataPeriod = dataPeriod
    }

    public func scenario(_ id: String) -> ChronosScenario? { scenarios.first { $0.id == id } }

    public func time(at step: Int, scenarioID: String? = nil) -> Date {
        let base = scenarioID.flatMap { scenario($0)?.origin } ?? origin
        return base.addingTimeInterval(Double(step * intervalSeconds))
    }

    public func history(scenarioID: String, areaID: String, step: Int) -> [Float]? {
        if let scenario = scenario(scenarioID), let history = scenario.apiHistory {
            guard let snapshot = try? ForecastInputSnapshot(history: history, actuals: scenario.apiActuals, step: step)
            else { return nil }
            return snapshot.series.first { $0.areaID == areaID }?.values.map { $0 ?? .nan }
        }
        guard let scenario = scenario(scenarioID), let history = scenario.history[areaID],
            let truth = scenario.truth[areaID]
        else { return nil }
        return ChronosHistoryWindow.make(history: history, truth: truth, revealed: step, maxLength: historySteps)
    }

    public func request(scenarioID: String, step: Int, mode: ChronosForecastMode) -> ChronosForecastRequest? {
        guard step >= 0, step <= truthSteps - horizon, scenario(scenarioID) != nil else { return nil }
        if let scenario = scenario(scenarioID), let history = scenario.apiHistory {
            guard mode == .live,
                let snapshot = try? ForecastInputSnapshot(history: history, actuals: scenario.apiActuals, step: step)
            else { return nil }
            return ChronosForecastRequest(
                scenarioID: scenarioID, originStep: step, mode: .live,
                series: snapshot.series.map {
                    ChronosSeriesInput(areaID: $0.areaID, values: $0.values.map { $0 ?? .nan })
                },
                snapshot: snapshot)
        }
        var inputs: [ChronosSeriesInput] = []
        for area in areas {
            guard let values = history(scenarioID: scenarioID, areaID: area.id, step: step) else { return nil }
            inputs.append(ChronosSeriesInput(areaID: area.id, values: values))
        }
        return ChronosForecastRequest(scenarioID: scenarioID, originStep: step, mode: mode, series: inputs)
    }

    public func revealedValue(scenarioID: String, areaID: String, targetStep: Int, clockStep: Int) -> Float? {
        if let scenario = scenario(scenarioID), scenario.apiHistory != nil {
            guard targetStep >= 0, targetStep < clockStep,
                let values = scenario.apiActuals?.series.first(where: { $0.areaID == areaID })?.values,
                values.indices.contains(targetStep)
            else { return nil }
            return values[targetStep]
        }
        guard targetStep >= 0, targetStep < clockStep, let values = scenario(scenarioID)?.truth[areaID],
            values.indices.contains(targetStep)
        else { return nil }
        return values[targetStep]
    }
}

public struct ChronosLoadedDemo: Equatable, Sendable {
    public let catalog: ChronosCatalog
    public let liveAvailable: Bool

    public init(catalog: ChronosCatalog, liveAvailable: Bool) {
        self.catalog = catalog
        self.liveAvailable = liveAvailable
    }
}
