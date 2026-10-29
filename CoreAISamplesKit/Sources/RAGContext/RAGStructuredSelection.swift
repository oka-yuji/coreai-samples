import Foundation

public struct RAGCandidate: Sendable {
    public var source: RAGSource
    public var headers: [RAGSource]
    public var rankingText: String
    public var sectionID: String
    public var sectionTitle: String
    public var structured: Bool
}

public struct RAGCandidateSet: Sendable {
    public var candidates: [RAGCandidate]
    public var fallbackReason: String?
}

public struct RAGSelection: Sendable {
    public var candidates: [RAGCandidate]
    public var sources: [RAGSource]
    public var hasMatches: Bool
}

public enum RAGStructuredSelection {
    public static func candidates(_ document: RAGDocumentSnapshot) -> RAGCandidateSet {
        guard let structure = document.structure, structure.isAvailable else {
            return flat(document, reason: document.structure?.unavailableReason ?? "structureNotCaptured")
        }
        guard Set(structure.sections.map(\.id)).count == structure.sections.count else {
            return flat(document, reason: "invalidStructure")
        }
        let sections = Dictionary(structure.sections.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        if structure.sections.contains(where: { section in
            section.body.text(in: document.text)?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == true
                && !structure.sections.contains(where: { $0.parentID == section.id })
        }) {
            return flat(document, reason: "headingOnlyLeaf")
        }
        var candidates: [RAGCandidate] = []
        if let range = structure.preamble {
            candidates += body(
                document, range: range, headers: [], title: document.title,
                sectionID: RAGContext.fingerprint(document.id + "\n" + document.version) + "-preamble")
        }
        for section in structure.sections {
            let path = section.ancestorIDs + [section.id]
            guard path.allSatisfy({ sections[$0] != nil }) else { return flat(document, reason: "invalidStructure") }
            let ancestors = path.compactMap { sections[$0] }
            let headers = ancestors.compactMap { source(document, range: $0.heading) }
            guard headers.count == ancestors.count, section.body.text(in: document.text) != nil else {
                return flat(document, reason: "invalidStructure")
            }
            let title = ([document.title] + ancestors.map(\.title)).joined(separator: " / ")
            candidates += body(document, range: section.body, headers: headers, title: title, sectionID: section.id)
        }
        return candidates.isEmpty ? flat(document, reason: "noBodyCandidates") : RAGCandidateSet(candidates: candidates)
    }

    public static func select(
        _ candidates: [RAGCandidate], question: String, budget: Int, count: (String) -> Int,
        rank: ([String], String) -> [Int]
    ) -> RAGSelection {
        let raw = rank(candidates.map(\.rankingText), question)
        var seenRanks: Set<Int> = []
        let ranked = raw.filter { candidates.indices.contains($0) && seenRanks.insert($0).inserted }
        guard budget > 0 else { return RAGSelection(candidates: [], sources: [], hasMatches: !ranked.isEmpty) }
        let order = ranked + candidates.indices.filter { !seenRanks.contains($0) }
        var seen: Set<String> = []
        var sources: [RAGSource] = []
        var picked: [RAGCandidate] = []
        var used = 0
        for index in order {
            let candidate = candidates[index]
            guard !seen.contains(identity(candidate.source)) else { continue }
            var proposed = seen
            let additional = (candidate.headers + [candidate.source]).filter { proposed.insert(identity($0)).inserted }
            let cost = additional.reduce(0) { $0 + count(RAGContext.measuredBlock($1)) + 2 }
            guard cost >= 0, cost <= budget - used else { continue }
            sources += additional
            picked.append(candidate)
            seen = proposed
            used += cost
        }
        return RAGSelection(candidates: picked, sources: sources, hasMatches: !ranked.isEmpty)
    }

    public static func identity(_ source: RAGSource) -> String {
        [
            source.document ?? source.url, source.version ?? "", String(source.offset ?? -1),
            RAGContext.fingerprint(source.text),
        ]
        .joined(separator: "\n")
    }

    public static func uniqueSources(_ sources: [RAGSource]) -> [RAGSource] {
        var seen: Set<String> = []
        return sources.filter { seen.insert(identity($0)).inserted }
    }

    private static func flat(_ document: RAGDocumentSnapshot, reason: String) -> RAGCandidateSet {
        let candidates = document.sources.map {
            RAGCandidate(
                source: $0, headers: [], rankingText: $0.text,
                sectionID: RAGContext.fingerprint(document.id + "\n" + document.version) + "-flat-"
                    + String($0.offset ?? 0),
                sectionTitle: document.title, structured: false)
        }
        return RAGCandidateSet(candidates: candidates, fallbackReason: reason)
    }

    private static func body(
        _ document: RAGDocumentSnapshot, range: RAGTextRange, headers: [RAGSource], title: String, sectionID: String
    ) -> [RAGCandidate] {
        guard let text = range.text(in: document.text), !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return []
        }
        return RAGContext.documentSources(
            title: document.title, url: document.url, document: document.id,
            date: document.date, text: text
        ).map {
            var fragment = $0
            fragment.offset = range.lowerBound + ($0.offset ?? 0)
            fragment.version = document.version
            return RAGCandidate(
                source: fragment, headers: headers, rankingText: title + "\n" + fragment.text,
                sectionID: sectionID, sectionTitle: title, structured: true)
        }
    }

    private static func source(_ document: RAGDocumentSnapshot, range: RAGTextRange) -> RAGSource? {
        guard let text = range.text(in: document.text), !text.isEmpty else { return nil }
        var source = RAGSource(title: document.title, url: document.url, date: document.date, text: text)
        source.document = document.id
        source.version = document.version
        source.offset = range.lowerBound
        return source
    }
}
