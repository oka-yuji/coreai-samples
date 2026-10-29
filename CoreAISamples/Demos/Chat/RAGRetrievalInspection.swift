import Foundation
import RAGContext

@MainActor
enum RAGRetrievalInspection {
    struct Row: Encodable {
        var caseID: String
        var requestedStrategy: String
        var effectiveStrategy: String
        var budgetUnit = "characters"
        var budget: Int
        var sources: [RAGEvaluationPassage]
        var bodies: [RAGEvaluationPassage]
        var candidates: [RAGEvaluationPassage]
        var fallbackReasons: [String]
        var retrievalSeconds: Double
    }

    static func inspect(_ manifest: RAGEvaluationManifest, strategy: RAGStrategy, output: URL) async throws {
        var rows: [Row] = []
        for item in manifest.cases {
            try Task.checkCancellation()
            let replay = try RAGEvaluationReplay(manifest: manifest, item: item)
            var settings = RAGSettings()
            settings.web = replay.documents.contains(where: \.isWeb)
            settings.local = replay.documents.contains { !$0.isWeb }
            settings.today = manifest.settings.today
            settings.sourceTokenBudget = manifest.settings.sourceTokenBudget
            settings.pageCharacterBudget = manifest.settings.pageCharacterBudget
            settings.pagesToRead = manifest.settings.pagesToRead
            settings.searchResults = manifest.settings.searchResults
            settings.strategy = strategy
            settings.recordsDiagnostics = true
            let result = await RAGRetriever.gather(
                question: item.question, settings: settings, count: { $0.count }, inputs: replay.inputs)
            try Task.checkCancellation()
            rows.append(
                Row(
                    caseID: item.id, requestedStrategy: strategy.rawValue, effectiveStrategy: result.effectiveStrategy,
                    budget: settings.sourceTokenBudget, sources: result.sources.compactMap(passage),
                    bodies: result.bodySources.compactMap(passage),
                    candidates: result.candidatePassages.compactMap(passage),
                    fallbackReasons: result.fallbacks, retrievalSeconds: result.seconds))
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(rows).write(to: output, options: .withoutOverwriting)
    }

    private static func passage(_ source: RAGSource) -> RAGEvaluationPassage? {
        guard let document = source.document, let version = source.version, let offset = source.offset else {
            return nil
        }
        return RAGEvaluationPassage(document: document, version: version, offset: offset, text: source.text)
    }
}
