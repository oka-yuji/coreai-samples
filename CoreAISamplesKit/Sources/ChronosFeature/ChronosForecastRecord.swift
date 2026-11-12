import ChronosCore
import Foundation

public struct ChronosInputCapture: Codable, Equatable, Sendable {
    public enum Source: String, Codable, Sendable { case api, fixed }
    public let source: Source
    public let scenarioID: String
    public let originStep: Int
    public let start: Date
    public let origin: Date
    public let series: [CountSeries]
    public let preprocessing: String
    public let api: ForecastInputSnapshot?
    public let fixedFiles: [String: Data]

    public init(
        source: Source, scenarioID: String, originStep: Int, origin: Date, series: [CountSeries],
        preprocessing: String, api: ForecastInputSnapshot? = nil, fixedFiles: [String: Data] = [:]
    ) {
        self.source = source
        self.scenarioID = scenarioID
        self.originStep = originStep
        start = origin.addingTimeInterval(-1008 * 1800)
        self.origin = origin
        self.series = series
        self.preprocessing = preprocessing
        self.api = api
        self.fixedFiles = fixedFiles
    }
}

public struct ChronosInputEvidence: Codable, Equatable, Sendable {
    public let snapshot: ChronosInputCapture
    public let bytes: Data
    public let sha256: String

    public init(snapshot: ChronosInputCapture, bytes: Data, sha256: String) {
        self.snapshot = snapshot
        self.bytes = bytes
        self.sha256 = sha256
    }
}

public struct ChronosRecordDevice: Codable, Equatable, Sendable {
    public let hardwareModel: String
    public let operatingSystem: String
    public let operatingSystemVersion: String
    public let power: String
    public let thermal: String
    public let memory: String

    public init(hardwareModel: String, operatingSystem: String, operatingSystemVersion: String) {
        self.hardwareModel = hardwareModel
        self.operatingSystem = operatingSystem
        self.operatingSystemVersion = operatingSystemVersion
        power = "unknown"
        thermal = "unknown"
        memory = "unknown"
    }
}

public struct ChronosRecordedArea: Codable, Equatable, Sendable {
    public let areaID: String
    public let identity: ChronosBundleIdentity
    public let raw: ChronosPrediction

    public init(areaID: String, identity: ChronosBundleIdentity, raw: ChronosPrediction) {
        self.areaID = areaID
        self.identity = identity
        self.raw = raw
    }

    public var forecast: ChronosAreaForecast { ChronosAreaForecast(areaID: areaID, raw: raw, identity: identity) }
}

public struct ChronosForecastRecord: Codable, Equatable, Identifiable, Sendable {
    public let schema: String
    public let id: UUID
    public let savedAt: Date
    public let mode: ChronosForecastMode
    public let identity: ChronosBundleIdentity
    public let quantileLevels: [Double]
    public let input: ChronosInputEvidence
    public let inputSHA256: String?
    public let areas: [ChronosRecordedArea]
    public let device: ChronosRecordDevice
    public let processingSeconds: Double
    public let timingBoundary: String
    public let timestampAuthority: String

    public init(id: UUID, savedAt: Date, batch: ChronosForecastBatch, device: ChronosRecordDevice) throws {
        guard batch.mode == .live, let identity = batch.identity, let input = batch.capturedInput,
            let seconds = batch.processingSeconds, seconds.isFinite, seconds >= 0,
            batch.scenarioID == input.snapshot.scenarioID, batch.originStep == input.snapshot.originStep,
            batch.inputSnapshot == input.snapshot.api,
            batch.areas.count == 20, batch.areas.map(\.areaID) == input.snapshot.series.map(\.areaID)
        else { throw ChronosDemoFailure.invalidData }
        areas = try batch.areas.map {
            guard let raw = $0.raw, $0.identity == identity else { throw ChronosDemoFailure.invalidData }
            return ChronosRecordedArea(areaID: $0.areaID, identity: identity, raw: raw)
        }
        schema = "chronos-native-record/1"
        self.id = id
        self.savedAt = savedAt
        mode = .live
        self.identity = identity
        quantileLevels = ChronosContract.quantileLevels
        self.input = input
        inputSHA256 = batch.inputSHA256
        self.device = device
        processingSeconds = seconds
        timingBoundary = "sequential-native-predictions-including-preparation"
        timestampAuthority = "local-device-clock-not-certified"
    }
}

public struct ChronosActualCheck: Codable, Equatable, Identifiable, Sendable {
    public let schema: String
    public let id: UUID
    public let recordID: UUID
    public let checkedAt: Date
    public let actuals: APIWindow
    public let metrics: ChronosForecastMetrics

    public init(id: UUID, record: ChronosForecastRecord, checkedAt: Date, actuals: APIWindow) throws {
        guard actuals.start == record.input.snapshot.origin, actuals.count == 4,
            Set(actuals.series.map(\.areaID)) == Set(record.areas.map(\.areaID)), actuals.series.count == 20
        else { throw ChronosDemoFailure.invalidData }
        metrics = try ChronosForecastMetrics.evaluate(
            record.areas.map(\.forecast),
            actuals: Dictionary(
                uniqueKeysWithValues: actuals.series.map {
                    ($0.areaID, $0.values)
                }))
        schema = "chronos-actual-check/1"
        self.id = id
        recordID = record.id
        self.checkedAt = checkedAt
        self.actuals = actuals
    }

    public var status: ChronosCheckStatus {
        if metrics.comparedCount == metrics.expectedCount { return .compared }
        if metrics.comparedCount > 0 { return .partial }
        if actuals.availability.contains(.unverified) { return .unverified }
        if actuals.availability.contains(.ambiguousTime) { return .incomparable }
        return .publicationPending
    }

    public func missingPairs(_ reason: SlotAvailability) -> Int {
        actuals.availability.filter { $0 == reason }.count * actuals.series.count
    }
}

public enum ChronosCheckStatus: String, Equatable, Sendable {
    case notChecked, publicationPending, partial, compared, incomparable, unverified
}

public struct ChronosJournalEntry: Equatable, Identifiable, Sendable {
    public let id: UUID
    public let savedAt: Date?
    public let origin: Date?
    public let source: ChronosInputCapture.Source?

    public init(id: UUID, record: ChronosForecastRecord?) {
        self.id = id
        savedAt = record?.savedAt
        origin = record?.input.snapshot.origin
        source = record?.input.snapshot.source
    }
}

public struct ChronosJournalDetail: Equatable, Sendable {
    public let record: ChronosForecastRecord
    public let checks: [ChronosActualCheck]

    public init(record: ChronosForecastRecord, checks: [ChronosActualCheck]) {
        self.record = record
        self.checks = checks
    }
}
