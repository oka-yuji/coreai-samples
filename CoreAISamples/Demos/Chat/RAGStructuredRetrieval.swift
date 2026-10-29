import Foundation
import RAGContext

extension RAGRetriever {
    nonisolated static func applyStructured(
        to retrieval: inout RAGRetrieval, question: String, settings: RAGSettings,
        budget: Int, count: @Sendable (String) -> Int
    ) {
        retrieval.requestedStrategy = settings.strategy.rawValue
        guard settings.strategy != .flat else { return }
        var pool: [RAGCandidate] = []
        var hasStructure = false
        var hadPageMatches = false
        for document in retrieval.documents {
            let set = RAGStructuredSelection.candidates(document)
            hasStructure = hasStructure || set.candidates.contains(where: \.structured)
            if let reason = set.fallbackReason { retrieval.fallbacks.append(document.id + ": " + reason) }
            let scheme = URL(string: document.id)?.scheme?.lowercased()
            if scheme == "https" || scheme == "http" {
                let selected = RAGStructuredSelection.select(
                    set.candidates, question: question, budget: settings.pageCharacterBudget,
                    count: { $0.count }, rank: bestRanking)
                pool += selected.candidates
                hadPageMatches = hadPageMatches || selected.hasMatches
                if selected.candidates.isEmpty && !set.candidates.isEmpty {
                    retrieval.fallbacks.append(document.id + ": pageBudgetExceeded")
                }
            } else {
                pool += set.candidates
            }
        }
        if settings.strategy == .structureLLM {
            retrieval.selectionCandidates = pool
            retrieval.candidatePassages = pool.map(\.source)
        }
        guard hasStructure else {
            retrieval.fallbacks.append("noStructuredCandidates")
            return
        }
        if pool.isEmpty {
            retrieval.effectiveStrategy = RAGStrategy.structureBM25.rawValue
            retrieval.sources = []
            retrieval.bodySources = []
            retrieval.candidates = 0
            retrieval.candidatePassages = []
            return
        }
        let selected = RAGStructuredSelection.select(
            pool, question: question, budget: budget, count: count, rank: bestRanking)
        guard selected.hasMatches || hadPageMatches else {
            retrieval.fallbacks.append("noMatchingCandidates")
            return
        }
        if !selected.hasMatches && hadPageMatches {
            retrieval.fallbacks.append("matchingCandidatesExceedPageBudget")
        }
        retrieval.effectiveStrategy = RAGStrategy.structureBM25.rawValue
        retrieval.sources = selected.sources
        retrieval.atomicGroups = selected.candidates.map { $0.headers + [$0.source] }
        retrieval.bodySources = selected.candidates.map(\.source)
        retrieval.candidates = pool.count
        if settings.recordsDiagnostics {
            retrieval.selectionCandidates = pool
            retrieval.candidatePassages = pool.map(\.source)
        }
    }
}
