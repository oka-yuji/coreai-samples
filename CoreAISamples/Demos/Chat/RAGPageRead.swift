import Foundation
import RAGContext

extension RAGRetriever {
    static func readPageDetails(
        url: URL, policy: AgentFetchPolicy = AgentFetchPolicy(),
        download: @Sendable (URL) async throws -> (text: String, finalURL: URL?, contentType: String) = AgentTools
            .downloadPage,
        render: @Sendable (URL) async throws -> (text: String, finalURL: URL?) = AgentPageReader.read
    ) async throws -> RAGPage {
        try await policy.validate(url)
        var result = RAGPage(raw: "", text: "", finalURL: url.absoluteString, format: .renderedText)
        do {
            let page = try await download(url)
            result.raw = page.text
            result.text = AgentTools.plainText(fromHTML: page.text)
            result.finalURL = page.finalURL?.absoluteString ?? url.absoluteString
            result.format = page.contentType == "text/plain" ? .text : .html
        } catch let failure as AgentToolMessage where failure.forbidsPageFallback {
            throw failure
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            AgentTools.log("[rag] \(url.absoluteString) direct read failed: \(error)")
        }
        if result.text.count < 400 {
            let page = try await render(url)
            result.text = AgentTools.condense(page.text)
            result.finalURL = page.finalURL?.absoluteString ?? url.absoluteString
            result.format = .renderedText
        }
        try Task.checkCancellation()
        return result
    }
}
