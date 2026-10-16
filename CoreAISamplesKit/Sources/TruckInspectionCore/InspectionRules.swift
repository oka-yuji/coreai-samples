import Foundation

public enum InspectionAnswerMethod: String, Codable, Sendable {
    case button, exact, model
}

public struct InspectionCandidate: Codable, Equatable, Sendable {
    public let status: InspectionStatus
    public let method: InspectionAnswerMethod
    public let score: Double?

    public init(status: InspectionStatus, method: InspectionAnswerMethod, score: Double? = nil) {
        self.status = status
        self.method = method
        self.score = score
    }

    public var needsConfirmation: Bool { method == .model }
}

public enum InspectionConfirmation: Equatable, Sendable {
    case yes, no, unknown
}

public enum InspectionRules {
    public static let answerCharacterLimit = 240

    public static func normalize(_ text: String) -> String {
        let ignored = CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters)
        let folded = text.folding(options: [.widthInsensitive, .caseInsensitive], locale: Locale(identifier: "ja_JP"))
        return String(folded.unicodeScalars.filter { !ignored.contains($0) })
    }

    public static func exactAnswer(
        _ text: String, item: InspectionItem, catalog: InspectionCatalog
    ) throws -> InspectionCandidate? {
        let value = normalize(text)
        guard !value.isEmpty else { throw InspectionError.emptyAnswer }
        guard text.count <= answerCharacterLimit else { throw InspectionError.answerTooLong }
        let matches = catalog.choices.filter { choice in
            let words = [choice.title] + choice.aliases + (item.aliases[choice.status.rawValue] ?? [])
            return words.contains { normalize($0) == value }
        }
        guard matches.count == 1, let match = matches.first else { return nil }
        return InspectionCandidate(status: match.status, method: .exact)
    }

    public static func classify(_ scores: [Float], choices: [InspectionChoice]) throws -> InspectionCandidate {
        guard scores.count == choices.count, !scores.isEmpty,
            scores.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
            let best = scores.max(), best > 0, scores.filter({ $0 == best }).count == 1,
            let index = scores.firstIndex(of: best)
        else { throw InspectionError.invalidClassification }
        return InspectionCandidate(status: choices[index].status, method: .model, score: Double(best))
    }

    public static func confirmation(_ text: String, vocabulary: InspectionVocabulary) -> InspectionConfirmation {
        let value = normalize(text)
        guard !value.isEmpty else { return .unknown }
        if vocabulary.no.contains(where: { normalize($0) == value }) { return .no }
        if vocabulary.yes.contains(where: { normalize($0) == value }) { return .yes }
        return .unknown
    }

    public static func resolveFinal(_ text: String?, lastPartial: String) -> String {
        let value = (text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty ? lastPartial.trimmingCharacters(in: .whitespacesAndNewlines) : value
    }

    public static func speechDeadline(for text: String) -> Double {
        max(8, 3 + 0.35 * Double(text.count))
    }
}

public struct InspectionListeningWindow: Equatable, Sendable {
    public enum Decision: Equatable, Sendable { case keepListening, endAudio, resolve }
    public private(set) var lastPartial = ""
    public private(set) var lastChange: Double?
    public private(set) var endedAt: Double?

    public init() {}

    public mutating func partial(_ text: String, elapsed: Double) {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, value != lastPartial else { return }
        lastPartial = value
        lastChange = elapsed
    }

    public mutating func tick(elapsed: Double) -> Decision {
        if let endedAt { return elapsed - endedAt >= 2 ? .resolve : .keepListening }
        if elapsed >= 6 || lastChange.map({ elapsed - $0 >= 1.2 }) == true {
            endedAt = elapsed
            return .endAudio
        }
        return .keepListening
    }
}
