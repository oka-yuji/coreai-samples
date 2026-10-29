import Foundation

public enum RAGStrategy: String, Codable, Sendable, CaseIterable {
    case flat
    case structureBM25 = "structure-bm25"
    case structureLLM = "structure-llm"
}

public struct RAGEvaluationDocument: Codable, Equatable, Sendable {
    public var id: String
    public var url: String
    public var title: String
    public var date: String
    public var format: String
    public var raw: String
    public var text: String
    public var sha256: String

    public var isWeb: Bool { url.hasPrefix("https://") }

    public var snapshot: RAGDocumentSnapshot {
        RAGDocumentSnapshot(
            id: url, url: isWeb ? url : (URL(string: url)?.lastPathComponent ?? title), finalURL: url,
            title: title, date: date, raw: raw, text: text, format: RAGDocumentFormat(rawValue: format) ?? .text)
    }

    public var sources: [RAGSource] { snapshot.sources }

}

public struct RAGEvaluationEvidence: Codable, Equatable, Sendable {
    public var documentID: String
    public var version: String
    public var offset: Int
    public var text: String
}

public struct RAGEvaluationCase: Codable, Equatable, Sendable {
    public var id: String
    public var split: String
    public var category: String
    public var language: String
    public var question: String
    public var documentIDs: [String]
    public var gold: [RAGEvaluationEvidence]
    public var expectedPoints: [String]
    public var expectsAbstention: Bool
}

public struct RAGEvaluationSettings: Codable, Equatable, Sendable {
    public var modelID: String
    public var expectedModelHash: String?
    public var today: String
    public var sourceTokenBudget: Int
    public var pageCharacterBudget: Int
    public var searchResults: Int
    public var pagesToRead: Int
    public var maxSteps: Int
    public var maxTokens: Int
    public var contextLimit: Int
    public var effort: String
    public var citationGuard: Bool
    public var deadlineSeconds: Double
}

public struct RAGEvaluationManifest: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var datasetID: String
    public var parserVersion: String
    public var offsetUnit: String
    public var settings: RAGEvaluationSettings
    public var documents: [RAGEvaluationDocument]
    public var cases: [RAGEvaluationCase]

    public enum Failure: Error, Equatable, LocalizedError {
        case invalid(String)

        public var errorDescription: String? {
            switch self {
            case .invalid(let reason): "Invalid RAG evaluation input: " + reason
            }
        }
    }

    public func validate(strategy: String = "flat", requiresModelPin: Bool = false) throws {
        guard strategy == RAGStrategy.flat.rawValue else { throw Failure.invalid("unsupported strategy: " + strategy) }
        guard schemaVersion == 1, parserVersion == "flat-v1", offsetUnit == "swift-character",
            !datasetID.isEmpty, !documents.isEmpty, !cases.isEmpty
        else { throw Failure.invalid("schema, parser, offset unit or empty dataset") }
        guard settings.modelID == "Qwen/Qwen3.8-27B",
            [3_000, 6_000].contains(settings.sourceTokenBudget),
            (500...10_000).contains(settings.pageCharacterBudget),
            (1...8).contains(settings.searchResults), (1...5).contains(settings.pagesToRead),
            (1...8).contains(settings.maxSteps), (1...8_192).contains(settings.maxTokens),
            settings.contextLimit == 16_384, ["off", "low"].contains(settings.effort),
            settings.deadlineSeconds.isFinite, (1...3_600).contains(settings.deadlineSeconds),
            RAGContext.isoDate(in: settings.today) == settings.today
        else { throw Failure.invalid("settings") }
        if requiresModelPin, settings.expectedModelHash?.isEmpty != false {
            throw Failure.invalid("freeze the verified Qwen3.8 4bit main.hash before model evaluation")
        }
        guard Set(documents.map(\.id)).count == documents.count,
            Set(documents.map(\.url)).count == documents.count,
            Set(cases.map(\.id)).count == cases.count,
            Set(cases.map(\.question)).count == cases.count
        else { throw Failure.invalid("duplicate identity or question") }
        for document in documents {
            guard !document.id.isEmpty, !document.title.isEmpty, !document.text.isEmpty,
                ["html", "markdown", "text"].contains(document.format),
                let url = URL(string: document.url),
                url.scheme == "https" && url.host != nil || url.scheme == "file",
                document.sha256 == RAGContext.fingerprint(document.text)
            else { throw Failure.invalid("document: " + document.id) }
        }
        for item in cases { try validate(item) }
    }

    public func selectedDocuments(for item: RAGEvaluationCase) throws -> [RAGEvaluationDocument] {
        try item.documentIDs.map { id in
            guard let document = documents.first(where: { $0.id == id }) else {
                throw Failure.invalid("unknown document: " + id)
            }
            return document
        }
    }

    private func validate(_ item: RAGEvaluationCase) throws {
        guard !item.id.isEmpty, ["development", "evaluation"].contains(item.split),
            ["heading", "same-title", "paraphrase", "multi-section", "flat", "absent"].contains(item.category),
            ["ja", "en"].contains(item.language), !item.question.isEmpty,
            !item.documentIDs.isEmpty, Set(item.documentIDs).count == item.documentIDs.count,
            item.expectsAbstention == item.gold.isEmpty,
            !item.expectedPoints.isEmpty
        else { throw Failure.invalid("case: " + item.id) }
        let available = try selectedDocuments(for: item)
        for evidence in item.gold {
            guard let document = available.first(where: { $0.id == evidence.documentID }),
                evidence.version == document.sha256, evidence.offset >= 0, !evidence.text.isEmpty,
                evidence.offset <= document.text.count,
                evidence.text.count <= document.text.count - evidence.offset
            else { throw Failure.invalid("evidence bounds: " + item.id) }
            let start = document.text.index(document.text.startIndex, offsetBy: evidence.offset)
            let end = document.text.index(start, offsetBy: evidence.text.count)
            guard document.text[start..<end].utf8.elementsEqual(evidence.text.utf8) else {
                throw Failure.invalid("evidence text: " + item.id)
            }
        }
    }
}
