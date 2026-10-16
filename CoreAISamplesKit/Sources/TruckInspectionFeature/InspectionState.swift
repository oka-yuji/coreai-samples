import Foundation
import TruckInspectionCore

public struct InspectionModelInfo: Equatable, Sendable {
    public let folder: URL
    public let revision: String
    public let milliseconds: Double
    public let computePreference: String
    public let floatType: String

    public init(
        folder: URL, revision: String, milliseconds: Double, computePreference: String, floatType: String = "unknown"
    ) {
        self.folder = folder
        self.revision = revision
        self.milliseconds = milliseconds
        self.computePreference = computePreference
        self.floatType = floatType
    }
}

public struct InspectionRecord: Codable, Equatable, Sendable {
    public let itemID: String
    public let title: String
    public let answer: String
    public let candidate: InspectionCandidate
    public let confirmed: Bool
    public let milliseconds: Double
    public let turnMilliseconds: Double?

    public init(
        itemID: String, title: String, answer: String, candidate: InspectionCandidate,
        confirmed: Bool, milliseconds: Double, turnMilliseconds: Double? = nil
    ) {
        self.itemID = itemID
        self.title = title
        self.answer = answer
        self.candidate = candidate
        self.confirmed = confirmed
        self.milliseconds = milliseconds
        self.turnMilliseconds = turnMilliseconds
    }
}

public struct InspectionState: Equatable, Sendable {
    public enum Phase: String, Equatable, Sendable {
        case idle, preparing, answering, classifying, confirming, completed, stopping, authorizing, speaking, listening,
            measuring
    }

    public let catalog: InspectionCatalog
    public var profile: TruckProfile = .airBrake
    public var phase = Phase.idle
    public var active = false
    public var generation = 0
    public var modelFolder: URL?
    public var modelInfo: InspectionModelInfo?
    public var records: [String: InspectionRecord] = [:]
    public var index = 0
    public var draft = ""
    public var pending: InspectionCandidate?
    public var pendingMilliseconds: Double = 0
    public var measurement: InspectionMeasurement?
    public var measurementReturnPhase = Phase.idle
    public var message = ""
    public var events: [String] = []
    public var reloadAfterStop = false
    public var resumeAfterStop = Phase.idle
    public var voiceActive = false
    public var voicePrompt = InspectionPrompt.question
    public var spokenText = ""
    public var partialText = ""
    public var voiceDiagnostics = ""
    public var speechTimeouts = 0
    public var emptyAnswers = 0
    public var unclearConfirmations = 0
    public var clockSeconds: Double = 0
    public var turnStartedAt: Double?

    public init(catalog: InspectionCatalog, modelFolder: URL? = nil) {
        self.catalog = catalog
        self.modelFolder = modelFolder
    }

    public var items: [InspectionItem] { catalog.items(for: profile) }
    public var currentItem: InspectionItem? { items.indices.contains(index) ? items[index] : nil }
    public var canConfigure: Bool { active && (phase == .idle || phase == .completed) }
    public var canAnswer: Bool { active && phase == .answering }
    public var busy: Bool { phase == .preparing || phase == .classifying || phase == .stopping || phase == .measuring }
    public var orderedRecords: [InspectionRecord] { items.compactMap { records[$0.id] } }
    public var unansweredCount: Int { items.count - records.count }

    public var log: String {
        (["profile: \(profile.rawValue)", "state: \(phase.rawValue)"] + events).joined(separator: "\n")
    }

    public var report: String {
        var lines = [catalog.title, catalog.profiles[profile.rawValue] ?? profile.rawValue]
        for item in items {
            if let record = records[item.id] {
                let confirmation = record.confirmed ? "confirmed" : record.candidate.method.rawValue
                lines.append("\(item.title): \(catalog.title(for: record.candidate.status)) [\(confirmation)]")
                if !record.answer.isEmpty { lines.append("  \(record.answer)") }
            } else {
                lines.append("\(item.title): —")
            }
        }
        return lines.joined(separator: "\n")
    }

    public mutating func note(_ value: String) {
        events.append(value)
        if events.count > 1000 { events.removeFirst(events.count - 1000) }
    }
}
