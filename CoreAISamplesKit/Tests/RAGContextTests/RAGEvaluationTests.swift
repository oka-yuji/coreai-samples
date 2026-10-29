import Foundation
import Testing

@testable import RAGContext

@Suite("Fixed RAG evaluation")
struct RAGEvaluationTests {
    private func corpus() throws -> RAGEvaluationManifest {
        let root = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        return try JSONDecoder().decode(
            RAGEvaluationManifest.self, from: Data(contentsOf: root.appending(path: "Tools/RAGEvaluation/corpus.json")))
    }

    @Test("RG-E05 freezes separate development and evaluation questions with valid evidence")
    func frozenCorpus() throws {
        let manifest = try corpus()
        try manifest.validate()
        #expect(manifest.cases.filter { $0.split == "development" }.count == 12)
        let evaluation = manifest.cases.filter { $0.split == "evaluation" }
        #expect(evaluation.count == 24)
        for category in Set(evaluation.map(\.category)) {
            let cases = evaluation.filter { $0.category == category }
            #expect(cases.count == 4)
            #expect(cases.filter { $0.language == "ja" }.count == 3)
            #expect(cases.filter { $0.language == "en" }.count == 1)
        }
        #expect(manifest.settings.modelID == "Qwen/Qwen3.8-27B")
    }

    @Test(
        "RG-E03 rejects unimplemented strategies before execution",
        arguments: ["unknown"])
    func unsupported(_ strategy: String) throws {
        let manifest = try corpus()
        #expect(throws: RAGEvaluationManifest.Failure.self) { try manifest.validate(strategy: strategy) }
    }

    @Test("C requires frozen valid selection limits")
    func selectionLimits() throws {
        var manifest = try corpus()
        try manifest.validate(strategy: "structure-llm")
        manifest.settings.selection = nil
        #expect(throws: RAGEvaluationManifest.Failure.self) { try manifest.validate(strategy: "structure-llm") }
    }

    @Test("RG-E03 rejects changed documents, duplicate identities and invalid gold ranges", arguments: 0..<8)
    func invalidInputs(_ variant: Int) throws {
        var manifest = try corpus()
        switch variant {
        case 0: manifest.documents[0].text += "changed"
        case 1: manifest.cases[1].id = manifest.cases[0].id
        case 2: manifest.documents[1].url = manifest.documents[0].url
        case 3: manifest.cases[0].gold[0].offset = Int.max
        case 4: manifest.cases[0].gold[0].text = "invented evidence"
        case 5: manifest.cases[0].gold[0].version = "old"
        case 6: manifest.cases[0].documentIDs = ["missing"]
        default: manifest.settings.deadlineSeconds = .infinity
        }
        #expect(throws: RAGEvaluationManifest.Failure.self) { try manifest.validate() }
    }

    @Test("RG-E01 preserves flat source identity, offsets and packing order")
    func flatCompatibility() throws {
        let manifest = try corpus()
        let document = try #require(manifest.documents.first)
        let sources = document.sources
        var expected: [RAGSource] = []
        let characters = Array(document.text)
        for offset in stride(from: 0, to: characters.count, by: 400) {
            var source = RAGSource(
                title: document.title, url: document.url, date: document.date,
                text: String(characters[offset..<min(offset + 500, characters.count)]))
            source.document = document.url
            source.version = document.sha256
            source.offset = offset
            expected.append(source)
            if offset + 500 >= characters.count { break }
        }
        #expect(sources == expected)
        let rank: ([String], String) -> [Int] = { _, _ in [1, 0] }
        let ordered = [expected[1], expected[0]] + Array(expected.dropFirst(2))
        #expect(
            RAGContext.select(sources, question: "fixed", budget: 1_300, count: { $0.count }, rank: rank)
                == RAGContext.pack(ordered, budget: 1_300, count: { $0.count }))
    }

    @Test("RG-E03 uses Character offsets for combined Unicode and emoji")
    func unicodeEvidence() throws {
        var manifest = try corpus()
        let text = "Cafe\u{301} \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467} evidence"
        manifest.documents[0].text = text
        manifest.documents[0].raw = text
        manifest.documents[0].format = "text"
        manifest.documents[0].sha256 = RAGContext.fingerprint(text)
        var item = manifest.cases[0]
        item.gold = [
            RAGEvaluationEvidence(
                documentID: manifest.documents[0].id, version: manifest.documents[0].sha256,
                offset: 7, text: "evidence")
        ]
        manifest.cases = [item]
        try manifest.validate()
        manifest.cases[0].gold[0].offset = 8
        #expect(throws: RAGEvaluationManifest.Failure.self) { try manifest.validate() }
    }

    @Test("Evidence recovery requires the exact document version and complete original ranges")
    func evidenceRecovery() throws {
        let manifest = try corpus()
        let item = try #require(manifest.cases.first)
        let evidence = try #require(item.gold.first)
        let document = try #require(manifest.documents.first)
        let start = document.text.index(document.text.startIndex, offsetBy: evidence.offset)
        let middle = document.text.index(start, offsetBy: evidence.text.count / 2)
        let end = document.text.index(start, offsetBy: evidence.text.count)
        let first = RAGEvaluationPassage(
            document: document.url, version: document.sha256, offset: evidence.offset,
            text: String(document.text[start..<middle]))
        let last = RAGEvaluationPassage(
            document: document.url, version: document.sha256, offset: evidence.offset + first.text.count,
            text: String(document.text[middle..<end]))
        let both = RAGEvaluationMetrics.measure(item: item, manifest: manifest, passages: [first, last])
        #expect(both.recall == 1)
        #expect(both.hitAt1 == false)
        #expect(both.hitAt3 == true)
        var invalid = first
        invalid.text = String(repeating: "x", count: first.text.count)
        #expect(RAGEvaluationMetrics.measure(item: item, manifest: manifest, passages: [invalid, last]).recall == 0)
        invalid = first
        invalid.version = "another version"
        #expect(RAGEvaluationMetrics.measure(item: item, manifest: manifest, passages: [invalid, last]).recall == 0)
    }

    @Test("Questions with no evidence do not inflate evidence recall")
    func absentEvidence() throws {
        let manifest = try corpus()
        let absent = manifest.cases.first { $0.expectsAbstention }
        let item = try #require(absent)
        let metrics = RAGEvaluationMetrics.measure(item: item, manifest: manifest, passages: [])
        #expect(metrics.recall == nil)
        #expect(metrics.hitAt1 == nil)
        #expect(metrics.expectedEvidence == 0)
    }

    @Test("Canonical Unicode equivalence does not authorize rewriting original evidence")
    func rejectsNormalizedEvidence() throws {
        var manifest = try corpus()
        let original = "Cafe\u{301} evidence"
        let normalized = "Caf\u{E9} evidence"
        manifest.documents[0].text = original
        manifest.documents[0].raw = original
        manifest.documents[0].sha256 = RAGContext.fingerprint(original)
        manifest.cases = [manifest.cases[0]]
        manifest.cases[0].gold = [
            RAGEvaluationEvidence(
                documentID: manifest.documents[0].id, version: manifest.documents[0].sha256,
                offset: 0, text: normalized)
        ]
        #expect(throws: RAGEvaluationManifest.Failure.self) { try manifest.validate() }
        manifest.cases[0].gold[0].text = original
        try manifest.validate()
        let rewritten = RAGEvaluationPassage(
            document: manifest.documents[0].url, version: manifest.documents[0].sha256,
            offset: 0, text: normalized)
        #expect(
            RAGEvaluationMetrics.measure(item: manifest.cases[0], manifest: manifest, passages: [rewritten]).recall == 0
        )
    }

    @Test("Manifest fingerprints describe original bytes for every accepted JSON encoding")
    func manifestBytes() throws {
        let text = #"{"schemaVersion":1}"#
        let utf8 = Data(text.utf8)
        let utf16 = try #require(text.data(using: .utf16))
        let decoded = try JSONDecoder().decode([String: Int].self, from: utf16)
        #expect(decoded["schemaVersion"] == 1)
        #expect(RAGContext.fingerprint(utf8) == RAGContext.fingerprint(text))
        #expect(RAGContext.fingerprint(utf16) != RAGContext.fingerprint(text))
    }

    @Test("RG-E04 retains failures, cancellation, interruption and missing results in the denominator")
    func completeDenominator() throws {
        let expected = ["a", "b", "c", "d", "e"]
        let statuses: [RAGEvaluationStatus] = [.completed, .failed, .cancelled, .running]
        let results = zip(expected, statuses).map { id, status in
            var result = RAGEvaluationResult(caseID: id, manifestHash: "hash", strategy: "flat")
            result.status = status
            return result
        }
        let summary = try RAGEvaluationSummary.make(
            expected: expected, manifestHash: "hash", strategy: "flat", results: results)
        #expect(summary.requested == 5)
        #expect(summary.completed == 1 && summary.failed == 1 && summary.cancelled == 1)
        #expect(summary.interrupted == 1 && summary.notRun == 1 && summary.missing == ["e"])
        #expect(throws: RAGEvaluationManifest.Failure.self) {
            try RAGEvaluationSummary.make(expected: expected, manifestHash: "other", strategy: "flat", results: results)
        }
        #expect(throws: RAGEvaluationManifest.Failure.self) {
            try RAGEvaluationSummary.make(
                expected: expected, manifestHash: "hash", strategy: "flat", mode: "chat", results: results)
        }
    }
}
