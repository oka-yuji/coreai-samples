import Foundation

public enum RAGEvaluationStatus: String, Codable, Sendable, CaseIterable {
    case notRun
    case running
    case completed
    case failed
    case cancelled
}

public struct RAGEvaluationPassage: Codable, Equatable, Sendable {
    public var document: String
    public var version: String
    public var offset: Int
    public var text: String

    public init(document: String, version: String, offset: Int, text: String) {
        self.document = document
        self.version = version
        self.offset = offset
        self.text = text
    }
}

public struct RAGEvaluationMetrics: Codable, Equatable, Sendable {
    public var expectedEvidence: Int
    public var recoveredEvidence: Int
    public var hitAt1: Bool?
    public var hitAt3: Bool?
    public var recall: Double?

    public static func measure(
        item: RAGEvaluationCase, manifest: RAGEvaluationManifest, passages: [RAGEvaluationPassage]
    ) -> Self {
        let count = item.gold.filter { covered($0, manifest: manifest, passages: passages) }.count
        return Self(
            expectedEvidence: item.gold.count, recoveredEvidence: count,
            hitAt1: item.gold.isEmpty
                ? nil
                : item.gold.contains {
                    covered($0, manifest: manifest, passages: Array(passages.prefix(1)))
                },
            hitAt3: item.gold.isEmpty
                ? nil
                : item.gold.contains {
                    covered($0, manifest: manifest, passages: Array(passages.prefix(3)))
                }, recall: item.gold.isEmpty ? nil : Double(count) / Double(item.gold.count))
    }

    private static func covered(
        _ evidence: RAGEvaluationEvidence, manifest: RAGEvaluationManifest, passages: [RAGEvaluationPassage]
    ) -> Bool {
        guard let document = manifest.documents.first(where: { $0.id == evidence.documentID }) else { return false }
        let ranges = passages.filter {
            $0.document == document.url && $0.version == evidence.version
                && $0.offset >= 0 && $0.offset <= document.text.count
                && $0.text.count <= document.text.count - $0.offset
        }.filter {
            let start = document.text.index(document.text.startIndex, offsetBy: $0.offset)
            return document.text[start..<document.text.index(start, offsetBy: $0.text.count)].utf8.elementsEqual(
                $0.text.utf8)
        }.map { $0.offset..<($0.offset + $0.text.count) }.sorted { $0.lowerBound < $1.lowerBound }
        var end = evidence.offset
        for range in ranges where range.upperBound > end {
            if range.lowerBound > end { return false }
            end = range.upperBound
            if end >= evidence.offset + evidence.text.count { return true }
        }
        return false
    }
}

public struct RAGEvaluationResult: Codable, Sendable {
    public var schemaVersion = 1
    public var caseID: String
    public var mode = "work"
    public var manifestHash: String
    public var requestedStrategy: String
    public var effectiveStrategy: String?
    public var status: RAGEvaluationStatus = .notRun
    public var reason: String?
    public var finishReason: String?
    public var citationOutcome: String?
    public var answer = ""
    public var suppliedPassages: [RAGEvaluationPassage] = []
    public var candidates: [RAGEvaluationPassage] = []
    public var documents: [RAGDocumentSnapshot]?
    public var retrievalMetrics: RAGEvaluationMetrics?
    public var inputTokens: Int?
    public var reusedTokens: Int?
    public var outputTokens: Int?
    public var sourceTokens: Int?
    public var retrievalSeconds: Double?
    public var selectionSeconds: Double?
    public var answerTTFTSeconds: Double?
    public var totalSeconds: Double?
    public var lifetimePeakFootprintBytes: UInt64?
    public var memoryStatus = "notMeasured"
    public var sourceIDsValid: Bool?
    public var factualSupport: String?
    public var answerCorrectness: String?
    public var abstentionCorrectness: String?
    public var toolRequests: [String] = []
    public var replayErrors: [String] = []
    public var modelID: String?
    public var modelHash: String?
    public var tokenizerHash: String?
    public var osVersion: String?

    public init(caseID: String, manifestHash: String, strategy: String) {
        self.caseID = caseID
        self.manifestHash = manifestHash
        self.requestedStrategy = strategy
    }
}

public struct RAGEvaluationSummary: Codable, Equatable, Sendable {
    public var requested: Int
    public var completed: Int
    public var failed: Int
    public var cancelled: Int
    public var notRun: Int
    public var interrupted: Int
    public var missing: [String]

    public static func make(
        expected: [String], manifestHash: String, strategy: String, mode: String = "work",
        results: [RAGEvaluationResult]
    ) throws -> Self {
        guard Set(expected).count == expected.count, Set(results.map(\.caseID)).count == results.count,
            results.allSatisfy({
                expected.contains($0.caseID) && $0.manifestHash == manifestHash && $0.requestedStrategy == strategy
                    && $0.mode == mode
            })
        else { throw RAGEvaluationManifest.Failure.invalid("result identities") }
        let missing = expected.filter { id in !results.contains { $0.caseID == id } }
        return Self(
            requested: expected.count, completed: results.filter { $0.status == .completed }.count,
            failed: results.filter { $0.status == .failed }.count,
            cancelled: results.filter { $0.status == .cancelled }.count,
            notRun: results.filter { $0.status == .notRun }.count + missing.count,
            interrupted: results.filter { $0.status == .running }.count, missing: missing)
    }
}
