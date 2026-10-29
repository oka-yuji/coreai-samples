import Foundation
import RAGContext
import Synchronization

final class RAGEvaluationReplay: Sendable {
    struct Trace: Sendable {
        var requests: [String] = []
        var errors: [String] = []
    }

    let item: RAGEvaluationCase
    let documents: [RAGEvaluationDocument]
    let trace = Mutex(Trace())

    nonisolated init(manifest: RAGEvaluationManifest, item: RAGEvaluationCase) throws {
        self.item = item
        self.documents = try manifest.selectedDocuments(for: item)
        for document in documents {
            let parsed = document.format == "html" ? AgentTools.plainText(fromHTML: document.raw) : document.raw
            guard parsed.utf8.elementsEqual(document.text.utf8) else {
                throw RAGEvaluationManifest.Failure.invalid("normalization: " + document.id)
            }
        }
    }

    nonisolated var inputs: RAGInputs {
        RAGInputs(
            local: { [self] _ in documents.filter { !$0.isWeb }.sorted { $0.url < $1.url }.map(\.snapshot) },
            search: { [self] question, settings in
                try search(question, limit: settings.searchResults)
            },
            read: { [self] url in
                let document = try page(url.absoluteString)
                return RAGPage(
                    raw: document.raw, text: document.text, finalURL: document.url,
                    format: RAGDocumentFormat(rawValue: document.format) ?? .text)
            })
    }

    nonisolated func search(_ question: String, limit: Int) throws -> [AgentTools.SearchResult] {
        trace.withLock { $0.requests.append("search: " + question) }
        guard question == item.question else { throw miss("search: " + question) }
        return documents.filter(\.isWeb).prefix(limit).map {
            AgentTools.SearchResult(title: $0.title, url: $0.url, snippet: String($0.text.prefix(160)))
        }
    }

    nonisolated func page(_ url: String) throws -> RAGEvaluationDocument {
        trace.withLock { $0.requests.append("fetch: " + url) }
        guard let document = documents.first(where: { $0.isWeb && $0.url == url }) else {
            throw miss("fetch: " + url)
        }
        return document
    }

    nonisolated private func miss(_ request: String) -> AgentToolMessage {
        let message = "error: snapshot request not recorded: " + request
        trace.withLock { $0.errors.append(message) }
        return AgentToolMessage(text: message, fatal: true, forbidsPageFallback: true)
    }

    @MainActor
    func execute(_ call: AgentToolCall, settings: AgentToolSettings) async -> AgentToolOutput {
        trace.withLock { $0.requests.append("tool: " + call.name + "\n" + call.summary) }
        do {
            switch call.name {
            case "web_search":
                let results = try search(call.value("query") ?? "", limit: settings.searchResults)
                return AgentToolOutput(
                    text: results.map { "\($0.title)\n\($0.url)\n\($0.snippet)" }.joined(separator: "\n\n"))
            case "fetch_page":
                let document = try page(call.value("url") ?? "")
                let text = AgentTools.pageText(
                    document.text, query: call.value("query") ?? "", budget: settings.pageCharacterBudget)
                return AgentToolOutput(text: text, seen: [document.url: text], finalURL: document.url)
            case "write_note", "append_note", "move_note":
                return await AgentTools.run(call, settings: settings)
            default:
                return AgentToolOutput(text: "error: unsupported evaluation tool: " + call.name)
            }
        } catch let failure as AgentToolMessage {
            return AgentToolOutput(text: failure.text)
        } catch {
            return AgentToolOutput(text: "error: " + error.localizedDescription)
        }
    }
}
