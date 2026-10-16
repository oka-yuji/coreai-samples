import ConversationFeature
import ConversationRuntime
import Foundation
import Testing

struct ConversationArchiveTests {
    private let writer = UUID()
    @Test("Source identities and passages survive archive restore and missing source IDs are refused")
    func sourceArchive() async throws {
        let directory = folder()
        defer { try? FileManager.default.removeItem(at: directory) }
        var snapshot = sample()
        var sources = ConversationSources()
        let ids = try sources.register([
            ConversationSource(
                document: "file:///manual", version: String(repeating: "a", count: 64), title: "Manual",
                passages: [SourcePassage(offset: 30, text: "Selected passage")])
        ])
        snapshot.evidence = ConversationEvidence(sources: sources)
        snapshot.turns[0].sourceIDs = ids
        let archive = ConversationArchive(directory: directory)
        try await archive.save(snapshot, expectedRevision: 0, writer: writer)
        #expect(try await archive.load(id: snapshot.id) == .snapshot(snapshot))
        snapshot.turns[0].sourceIDs = ["S2"]
        #expect(throws: ConversationArchive.Failure.self) { try ConversationArchive.validate(snapshot) }
    }
    @Test("A new archive instance restores every saved turn and the matching Markdown export")
    func roundTrip() async throws {
        let directory = folder()
        defer { try? FileManager.default.removeItem(at: directory) }
        let snapshot = sample()
        try await ConversationArchive(directory: directory).save(snapshot, expectedRevision: 0, writer: writer)
        let reopened = ConversationArchive(directory: directory)
        #expect(try await reopened.latest() == .snapshot(snapshot))
        let markdown = try String(contentsOf: directory.appending(path: "sample.md"), encoding: .utf8)
        #expect(markdown == snapshot.markdown())
        let json = try String(contentsOf: directory.appending(path: "sample.json"), encoding: .utf8)
        #expect(!json.contains("braveAPIKey") && !json.contains("apiKey"))
        var next = snapshot
        next.revision += 1
        next.title = "Renamed"
        try await reopened.save(next, expectedRevision: snapshot.revision, writer: writer)
        await #expect(throws: ConversationArchive.Failure.self) {
            try await reopened.save(snapshot, expectedRevision: 0, writer: writer)
        }
        #expect(try await reopened.load(id: "sample") == .snapshot(next))
    }

    @Test(
        "A newer or malformed file opens its Markdown read-only and preserves the original JSON",
        arguments: [true, false])
    func unsupportedFile(newer: Bool) async throws {
        let directory = folder()
        defer { try? FileManager.default.removeItem(at: directory) }
        let archive = ConversationArchive(directory: directory)
        let original = sample()
        try await archive.save(original, expectedRevision: 0, writer: writer)
        var invalid = original
        if newer { invalid.formatVersion = 99 } else { invalid.turns[0].start = -1 }
        let data = try JSONEncoder().encode(invalid)
        let path = directory.appending(path: "sample.json")
        try data.write(to: path, options: .atomic)
        let loaded = try await archive.latest()
        guard case .readOnly(let id, let markdown, let reason) = loaded else {
            Issue.record("Expected a read-only Markdown fallback")
            return
        }
        #expect(id == "sample" && markdown == original.markdown() && !reason.isEmpty)
        #expect(try Data(contentsOf: path) == data)
    }

    @Test("A failed Markdown write leaves valid canonical JSON and reports failure until the export can be repaired")
    func failedExport() async throws {
        let directory = folder()
        defer { try? FileManager.default.removeItem(at: directory) }
        let archive = ConversationArchive(directory: directory)
        let snapshot = sample()
        let blocked = directory.appending(path: "sample.md")
        try FileManager.default.createDirectory(at: blocked, withIntermediateDirectories: true)
        await #expect(throws: (any Error).self) {
            try await archive.save(snapshot, expectedRevision: 0, writer: writer)
        }
        #expect(try await archive.load(id: "sample") == .snapshot(snapshot))
        try FileManager.default.removeItem(at: blocked)
        try await archive.save(snapshot, expectedRevision: 0, writer: writer)
        #expect(try String(contentsOf: blocked, encoding: .utf8) == snapshot.markdown())
    }

    @Test("An archive rejects path traversal and conflicting copies without overwriting existing history")
    func protectsFiles() async throws {
        let directory = folder()
        defer { try? FileManager.default.removeItem(at: directory) }
        let archive = ConversationArchive(directory: directory)
        let snapshot = sample()
        try await archive.save(snapshot, expectedRevision: 0, writer: writer)
        var changed = snapshot
        changed.title = "Conflicting copy"
        await #expect(throws: ConversationArchive.Failure.self) {
            try await archive.save(changed, expectedRevision: snapshot.revision, writer: writer)
        }
        changed.id = "../escape"
        await #expect(throws: ConversationArchive.Failure.self) {
            try await archive.save(changed, expectedRevision: snapshot.revision, writer: writer)
        }
        #expect(try await archive.load(id: snapshot.id) == .snapshot(snapshot))
    }

    @Test("Retrying a stale writer cannot overwrite another window and a separate copy preserves both histories")
    func concurrentWriters() async throws {
        let directory = folder()
        defer { try? FileManager.default.removeItem(at: directory) }
        let archive = ConversationArchive(directory: directory)
        let original = sample()
        try await archive.save(original, expectedRevision: 0, writer: writer)
        let otherWriter = UUID()
        var first = original
        first.revision += 1
        first.title = "Window A"
        try await archive.save(first, expectedRevision: original.revision, writer: writer)
        var second = original
        second.revision += 1
        second.title = "Window B"
        await #expect(throws: ConversationArchive.Failure.self) {
            try await archive.save(second, expectedRevision: original.revision, writer: otherWriter)
        }
        second.revision += 1
        await #expect(throws: ConversationArchive.Failure.self) {
            try await archive.save(second, expectedRevision: original.revision, writer: otherWriter)
        }
        #expect(try await archive.load(id: original.id) == .snapshot(first))
        second.id = "copy"
        second.revision = 1
        try await archive.save(second, expectedRevision: 0, writer: otherWriter)
        #expect(try await archive.load(id: "copy") == .snapshot(second))
        #expect(try await archive.load(id: original.id) == .snapshot(first))
    }

    @Test("Corrupt integer counters are rejected before restore or export can overflow")
    func invalidCounters() {
        var snapshot = sample()
        snapshot.revision = Int.max
        #expect(throws: ConversationArchive.Failure.self) { try ConversationArchive.validate(snapshot) }
        snapshot = sample()
        snapshot.turns[0].offset = Int.max
        #expect(throws: ConversationArchive.Failure.self) { try ConversationArchive.validate(snapshot) }
        snapshot = sample()
        snapshot.turns[0].steps = [
            StepStats(
                step: 1, ttftSeconds: nil, tokensPerSecond: nil, promptTokens: Int.max, reusedTokens: 0,
                generatedTokens: 1, toolCalls: 0, finish: "eos")
        ]
        #expect(throws: ConversationArchive.Failure.self) { try ConversationArchive.validate(snapshot) }
    }

    private func folder() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "conversation-archive-" + UUID().uuidString)
    }

    private func sample() -> ConversationSnapshot {
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        var state = ConversationState(
            id: "sample", now: date, modelID: "local-model", instructionsVersion: "frozen",
            toolsVersion: "web_search,fetch_page,write_note")
        _ = ConversationFeature.reduce(&state, .send("Remember the blue notebook", mode: .work, now: date))
        _ = ConversationFeature.reduce(
            &state,
            .transcriptChanged(
                run: 1,
                TranscriptRecord(entries: [
                    .instructions(id: "i", text: "Keep context"),
                    .prompt(id: "p", text: "Remember the blue notebook"),
                    .response(id: "a", text: "I will remember it.", metadata: StepMetadata(stopReason: "stopToken")),
                ])))
        _ = ConversationFeature.reduce(&state, .generationEnded(run: 1, stopReason: "stopToken"))
        return ConversationSnapshot(state)
    }
}
