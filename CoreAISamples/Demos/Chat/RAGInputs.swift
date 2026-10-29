import Foundation
import PDFKit
import RAGContext

struct RAGInputs: Sendable {
    var configure: @Sendable (RAGSettings) -> RAGSettings = { $0 }
    var local: @Sendable (URL) -> [RAGSource]
    var search: @Sendable (String, RAGSettings) async throws -> [AgentTools.SearchResult]
    var read: @Sendable (URL) async throws -> (html: String, text: String)

    nonisolated static var live: RAGInputs {
        RAGInputs(
            local: { directory in
                RAGDocuments.load(directory: directory) { PDFDocument(url: $0)?.string }
            },
            search: { question, settings in
                try await settings.braveAPIKey.isEmpty
                    ? AgentTools.duckDuckGoResults(query: question, limit: settings.searchResults)
                    : AgentTools.braveResults(query: question, limit: settings.searchResults, key: settings.braveAPIKey)
            }, read: { try await RAGRetriever.readPage(url: $0) })
    }
}
