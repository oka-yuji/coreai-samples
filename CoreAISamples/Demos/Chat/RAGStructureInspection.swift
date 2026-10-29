import Foundation
import RAGContext

@MainActor
enum RAGStructureInspection {
    static func inspect(_ manifest: RAGEvaluationManifest, output: URL) async throws {
        var collected: [String: RAGDocumentSnapshot] = [:]
        for item in manifest.cases {
            try Task.checkCancellation()
            let replay = try RAGEvaluationReplay(manifest: manifest, item: item)
            var settings = RAGSettings()
            settings.web = replay.documents.contains(where: \.isWeb)
            settings.local = replay.documents.contains { !$0.isWeb }
            settings.today = manifest.settings.today
            settings.captureStructure = true
            let result = await RAGRetriever.gather(
                question: item.question, settings: settings, count: { $0.count }, inputs: replay.inputs)
            try Task.checkCancellation()
            for document in result.documents {
                guard let expected = manifest.documents.first(where: { $0.url == document.id }),
                    expected.sha256 == document.version, expected.text.utf8.elementsEqual(document.text.utf8)
                else { throw RAGEvaluationManifest.Failure.invalid("structure changed the source") }
                if let previous = collected[document.id], previous != document {
                    throw RAGEvaluationManifest.Failure.invalid("structure depends on the query")
                }
                collected[document.id] = document
            }
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(collected.values.sorted { $0.id < $1.id }).write(to: output, options: .withoutOverwriting)
    }
}
