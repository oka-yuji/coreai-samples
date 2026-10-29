import Foundation
import RAGContext

@MainActor
enum RAGSelectionSelfTest {
    static func run(_ check: (String, Bool, String) -> Void) async {
        let raw =
            "<main><h1>Device manual</h1><h2>iPhone</h2><p>Wait seven minutes.</p><h2>Mac</h2><p>Wait thirty minutes.</p></main>"
        let text = AgentTools.plainText(fromHTML: raw)
        let page = RAGPage(raw: raw, text: text, finalURL: "https://rag.example/manual", format: .html)
        let inputs = RAGInputs(
            local: { _ in [] },
            search: { _, _ in [.init(title: "Devices", url: page.finalURL, snippet: "Devices")] },
            read: { _ in page })
        var settings = RAGSettings()
        settings.web = true
        settings.strategy = .structureBM25
        settings.recordsDiagnostics = true
        let selected = await RAGRetriever.gather(
            question: "Mac", settings: settings, count: { $0.count }, inputs: inputs)
        check(
            "rag-selection/parent-match",
            selected.effectiveStrategy == "structure-bm25"
                && selected.bodySources.first?.text.contains("thirty") == true
                && selected.sources.contains { $0.text == "Mac" }, "effective=\(selected.effectiveStrategy)")
        let cost = selected.sources.reduce(0) { $0 + RAGContext.measuredBlock($1).count + 2 }
        check("rag-selection/header-budget", cost <= settings.sourceTokenBudget, "used=\(cost)")
        let unmatched = await RAGRetriever.gather(
            question: "unmatchedword", settings: settings, count: { $0.count }, inputs: inputs)
        var flat = settings
        flat.strategy = .flat
        let baseline = await RAGRetriever.gather(
            question: "unmatchedword", settings: flat, count: { $0.count }, inputs: inputs)
        check(
            "rag-selection/no-match-fallback",
            unmatched.sources == baseline.sources
                && unmatched.effectiveStrategy == "flat" && unmatched.fallbacks.contains("noMatchingCandidates"),
            "fallbacks=\(unmatched.fallbacks)")
        var unsupported = inputs
        unsupported.read = { _ in RAGPage(raw: text, text: text, finalURL: page.finalURL, format: .renderedText) }
        let fallback = await RAGRetriever.gather(
            question: "Mac", settings: settings, count: { $0.count }, inputs: unsupported)
        check(
            "rag-selection/unsupported-fallback",
            fallback.effectiveStrategy == "flat"
                && fallback.fallbacks.contains("noStructuredCandidates"), "fallbacks=\(fallback.fallbacks)")
        let long =
            "<main><h1>" + String(repeating: "Mac ", count: 100)
            + "</h1><p>Too expensive.</p><h1>Other</h1><p>Small.</p></main>"
        var constrained = inputs
        constrained.read = { _ in
            RAGPage(raw: long, text: AgentTools.plainText(fromHTML: long), finalURL: page.finalURL, format: .html)
        }
        var limited = settings
        limited.pageCharacterBudget = 200
        let budgeted = await RAGRetriever.gather(
            question: "Mac", settings: limited, count: { $0.count }, inputs: constrained)
        check(
            "rag-selection/page-budget-reason",
            budgeted.effectiveStrategy == "structure-bm25"
                && budgeted.fallbacks.contains("matchingCandidatesExceedPageBudget")
                && !budgeted.fallbacks.contains("noMatchingCandidates"), "fallbacks=\(budgeted.fallbacks)")
        var denied = inputs
        denied.search = { _, _ in throw AgentToolMessage(text: "Unexpected web search") }
        denied.read = { _ in throw AgentToolMessage(text: "Unexpected page read") }
        var docsOnly = settings
        docsOnly.web = false
        docsOnly.local = true
        let noWeb = await RAGRetriever.gather(question: "Mac", settings: docsOnly, count: { $0.count }, inputs: denied)
        check(
            "rag-selection/docs-permission",
            noWeb.searchResults == 0 && noWeb.pagesRead == 0
                && !noWeb.notes.contains { $0.contains("Unexpected") }, "notes=\(noWeb.notes)")
    }
}
