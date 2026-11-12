import ChronosCore
import ChronosFeature
import Foundation

enum ChronosJournalValidation {
    static func input(_ evidence: ChronosInputEvidence, profile: NYCVerificationProfile? = nil) throws {
        let value = evidence.snapshot
        guard evidence.bytes.count <= 96 * 1024 * 1024,
            NYCAPIContract.hash(evidence.bytes) == evidence.sha256,
            (0...12).contains(value.originStep), NYCTime.isSupported(value.origin), NYCTime.isSupported(value.start),
            value.origin == NYCTime.aligned(value.origin),
            value.start == value.origin.addingTimeInterval(-1008 * 1800),
            value.series.count == 20, Set(value.series.map(\.areaID)) == Set(NYCAPIContract.zoneIDs),
            value.series.allSatisfy({
                $0.values.count == 1008 && $0.values.contains { $0 != nil }
                    && $0.values.allSatisfy {
                        $0.map { $0.isFinite && $0 >= 0 && $0 <= 16_777_216 && $0.rounded() == $0 } ?? true
                    }
            })
        else { throw ChronosDemoFailure.invalidData }
        switch value.source {
        case .fixed:
            try ChronosRepository.validateFixed(value)
        case .api:
            guard let api = value.api, value.fixedFiles.isEmpty,
                value.origin == api.origin, value.start == api.start, value.series == api.series,
                value.preprocessing == APIWindow.preprocessingVersion
            else { throw ChronosDemoFailure.invalidData }
            try NYCWindowValidation.validate(api.history, profile: profile)
            guard let originSeconds = Int(exactly: api.history.end.timeIntervalSince1970),
                value.scenarioID == "api-" + String(originSeconds)
            else { throw ChronosDemoFailure.invalidData }
            if let actuals = api.actuals { try NYCWindowValidation.validate(actuals, profile: profile) }
            guard try ForecastInputSnapshot(history: api.history, actuals: api.actuals, step: value.originStep) == api
            else { throw ChronosDemoFailure.invalidData }
        }
        guard try NYCAPIContract.encode(value) == evidence.bytes else { throw ChronosDemoFailure.invalidData }
    }

    static func record(_ value: ChronosForecastRecord, profile: NYCVerificationProfile? = nil) throws {
        guard value.schema == "chronos-native-record/1", value.mode == .live,
            NYCTime.isSupported(value.savedAt), value.processingSeconds.isFinite,
            value.processingSeconds >= 0,
            value.quantileLevels == ChronosContract.quantileLevels,
            value.timingBoundary == "sequential-native-predictions-including-preparation",
            value.timestampAuthority == "local-device-clock-not-certified",
            value.areas.count == 20, value.areas.map(\.areaID) == value.input.snapshot.series.map(\.areaID),
            value.areas.allSatisfy({ $0.identity == value.identity })
        else { throw ChronosDemoFailure.invalidData }
        try identity(value.identity)
        for area in value.areas { _ = try ChronosPrediction(quantiles: area.raw.quantiles) }
        let device = value.device
        guard
            device.hardwareModel.range(
                of: "^(Mac[0-9]+,[0-9]+|iPhone[0-9]+,[0-9]+|iPad[0-9]+,[0-9]+|unknown)$", options: .regularExpression)
                != nil,
            ["macOS", "iOS", "unknown"].contains(device.operatingSystem),
            device.operatingSystemVersion.range(
                of: "^[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}$", options: .regularExpression) != nil,
            device.power == "unknown", device.thermal == "unknown", device.memory == "unknown"
        else { throw ChronosDemoFailure.invalidData }
        try input(value.input, profile: profile)
        if let api = value.input.snapshot.api {
            guard value.inputSHA256 == NYCAPIContract.hash(try NYCAPIContract.encode(api)) else {
                throw ChronosDemoFailure.invalidData
            }
        } else if value.inputSHA256 != nil {
            throw ChronosDemoFailure.invalidData
        }
    }

    static func check(
        _ value: ChronosActualCheck, record: ChronosForecastRecord, profile: NYCVerificationProfile? = nil
    ) throws {
        guard value.schema == "chronos-actual-check/1", value.recordID == record.id,
            NYCTime.isSupported(value.checkedAt), value.actuals.start == record.input.snapshot.origin,
            value.actuals.count == 4
        else { throw ChronosDemoFailure.invalidData }
        try NYCWindowValidation.validate(value.actuals, profile: profile)
        guard
            try ChronosActualCheck(id: value.id, record: record, checkedAt: value.checkedAt, actuals: value.actuals)
                == value
        else { throw ChronosDemoFailure.invalidData }
    }

    private static func identity(_ identity: ChronosBundleIdentity) throws {
        guard identity.source.modelId == ChronosContract.source,
            identity.source.revision.range(of: "^[0-9a-f]{40}$", options: .regularExpression) != nil,
            Set(identity.files.keys)
                == Set([
                    "LICENSE", "NOTICE", "SOURCE_README.md", "reference.json", "verification.json",
                    "Model.aimodel/main.hash", "Model.aimodel/main.mlirb", "Model.aimodel/metadata.json",
                ])
        else { throw ChronosDemoFailure.invalidData }
        guard identity.files["Model.aimodel/main.mlirb"] != nil, identity.files.count <= 32,
            identity.files.keys.allSatisfy({
                !$0.hasPrefix("/") && !$0.contains("..") && !$0.contains("\\")
                    && $0.range(of: "^[A-Za-z0-9_./-]{1,128}$", options: .regularExpression) != nil
            })
        else { throw ChronosDemoFailure.invalidData }
        try identity.manifest.validate()
        for value in identity.files.values { try value.validate() }
    }
}
