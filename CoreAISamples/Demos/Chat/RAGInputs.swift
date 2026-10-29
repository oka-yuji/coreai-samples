import Foundation
import PDFKit
import RAGContext

struct RAGInputs: Sendable {
    var configure: @Sendable (RAGSettings) -> RAGSettings = { $0 }
    var local: @Sendable (URL) -> [RAGDocumentSnapshot]
    var search: @Sendable (String, RAGSettings) async throws -> [AgentTools.SearchResult]
    var read: @Sendable (URL) async throws -> RAGPage

    nonisolated static var live: RAGInputs {
        RAGInputs(
            local: { directory in
                RAGInputs.localDocuments(directory: directory)
            },
            search: { question, settings in
                try await settings.braveAPIKey.isEmpty
                    ? AgentTools.duckDuckGoResults(query: question, limit: settings.searchResults)
                    : AgentTools.braveResults(query: question, limit: settings.searchResults, key: settings.braveAPIKey)
            }, read: { try await RAGRetriever.readPageDetails(url: $0) })
    }
    nonisolated static func localDocuments(directory: URL) -> [RAGDocumentSnapshot] {
        let files =
            ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { RAGDocuments.extensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        return files.compactMap { url in
            guard !Task.isCancelled else { return nil }
            let kind = url.pathExtension.lowercased()
            let text = kind == "pdf" ? PDFDocument(url: url)?.string : try? String(contentsOf: url, encoding: .utf8)
            guard let text, !text.isEmpty else { return nil }
            let format: RAGDocumentFormat =
                kind == "pdf" ? .pdfText : ["md", "markdown"].contains(kind) ? .markdown : .text
            return RAGDocumentSnapshot(
                id: url.standardizedFileURL.absoluteString, url: url.lastPathComponent,
                finalURL: url.standardizedFileURL.absoluteString, title: url.lastPathComponent,
                date: RAGContext.isoDate(in: String(text.prefix(400))) ?? "", raw: text, text: text, format: format)
        }
    }
}

struct RAGPage: Sendable {
    var raw: String
    var text: String
    var finalURL: String
    var format: RAGDocumentFormat
}
