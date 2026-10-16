import ConversationFeature
import Foundation
import Testing

@testable import ConversationRuntime

struct ConversationLibraryTests {
    @Test("Renaming updates both files and the newest-first list while selection survives relaunch")
    func renameAndSelect() async throws {
        let folder = directory()
        defer { try? FileManager.default.removeItem(at: folder) }
        let archive = ConversationArchive(directory: folder)
        let writer = UUID()
        let first = snapshot("first", at: 100)
        let second = snapshot("second", at: 200)
        try await archive.save(first, expectedRevision: 0, writer: writer)
        try await archive.save(second, expectedRevision: 0, writer: writer)
        #expect(try await archive.list().map(\.id) == ["second", "first"])
        try await archive.select(id: "first")
        #expect(try await ConversationArchive(directory: folder).latest() == .snapshot(first))
        let result = try await archive.rename(
            id: "first", title: "Trip plan", expectedRevision: first.revision,
            now: Date(timeIntervalSince1970: 300), writer: writer)
        let renamed = result.snapshot
        #expect(result.exportError == nil)
        #expect(renamed.title == "Trip plan" && renamed.transcript == first.transcript)
        #expect(try await archive.list().map(\.title) == ["Trip plan", "second"])
        #expect(try await archive.load(id: "first") == .snapshot(renamed))
        #expect(
            try String(contentsOf: folder.appending(path: "first.md"), encoding: .utf8).contains("title: Trip plan"))
        await #expect(throws: ConversationArchive.Failure.self) {
            try await archive.rename(
                id: "first", title: "Old rename", expectedRevision: first.revision,
                now: Date(timeIntervalSince1970: 400), writer: UUID())
        }
        try await archive.select(id: nil)
        #expect(try await ConversationArchive(directory: folder).latest() == nil)
        #expect(try await archive.list().count == 2)
    }

    @Test("Deleting moves the canonical file and Markdown together to the actual macOS Trash")
    func movesToTrash() async throws {
        let folder = directory()
        defer { try? FileManager.default.removeItem(at: folder) }
        let archive = ConversationArchive(directory: folder)
        let value = snapshot("trash-test", at: 100)
        try await archive.save(value, expectedRevision: 0, writer: UUID())
        let trashed = try await archive.trash(id: value.id, expectedRevision: value.revision)
        defer { try? FileManager.default.removeItem(at: trashed) }
        #expect(trashed.path.contains(".Trash") && !trashed.lastPathComponent.hasPrefix("."))
        #expect(try await archive.list().isEmpty)
        #expect(!FileManager.default.fileExists(atPath: folder.appending(path: "trash-test.json").path))
        let json = try Data(contentsOf: trashed.appending(path: "trash-test.json"))
        #expect(try JSONDecoder().decode(ConversationSnapshot.self, from: json) == value)
        #expect(try String(contentsOf: trashed.appending(path: "trash-test.md"), encoding: .utf8) == value.markdown())
    }

    @Test("A failed Trash operation restores both original files and stale deletion cannot remove newer history")
    func failedDelete() async throws {
        let folder = directory()
        defer { try? FileManager.default.removeItem(at: folder) }
        let archive = ConversationArchive(directory: folder)
        let value = snapshot("preserved", at: 100)
        try await archive.save(value, expectedRevision: 0, writer: UUID())
        await #expect(throws: ConversationArchive.Failure.self) {
            try await archive.trash(
                id: value.id, expectedRevision: value.revision,
                move: { _ in
                    throw ConversationArchive.Failure.invalidSnapshot
                })
        }
        #expect(try await archive.load(id: value.id) == .snapshot(value))
        #expect(try String(contentsOf: folder.appending(path: "preserved.md"), encoding: .utf8) == value.markdown())
        await #expect(throws: ConversationArchive.Failure.self) {
            try await archive.trash(id: value.id, expectedRevision: value.revision - 1)
        }
        #expect(try await archive.list().map(\.id) == [value.id])
    }

    @Test("A rename keeps the committed JSON version visible when Markdown fails, and export retry changes no history")
    func renameExportFailure() async throws {
        let folder = directory()
        defer { try? FileManager.default.removeItem(at: folder) }
        let archive = ConversationArchive(directory: folder)
        let writer = UUID()
        let value = snapshot("rename-failure", at: 100)
        try await archive.save(value, expectedRevision: 0, writer: writer)
        let markdown = folder.appending(path: "rename-failure.md")
        try FileManager.default.removeItem(at: markdown)
        try FileManager.default.createDirectory(at: markdown, withIntermediateDirectories: false)
        let result = try await archive.rename(
            id: value.id, title: "Changed", expectedRevision: value.revision,
            now: Date(timeIntervalSince1970: 200), writer: writer)
        #expect(result.exportError != nil && result.snapshot.revision == value.revision + 1)
        #expect(try await archive.load(id: value.id) == .snapshot(result.snapshot))
        #expect(try await archive.list().first?.title == "Changed")
        try FileManager.default.removeItem(at: markdown)
        try await archive.repairExport(id: value.id)
        #expect(try String(contentsOf: markdown, encoding: .utf8) == result.snapshot.markdown())
        #expect(try await archive.load(id: value.id) == .snapshot(result.snapshot))
    }

    @Test(
        "A corrupt read-only conversation can be trashed without allowing a stale unknown-version delete of valid history"
    )
    func deletesCorruptFile() async throws {
        let folder = directory()
        defer { try? FileManager.default.removeItem(at: folder) }
        let archive = ConversationArchive(directory: folder)
        var value = snapshot("broken", at: 100)
        try await archive.save(value, expectedRevision: 0, writer: UUID())
        await #expect(throws: ConversationArchive.Failure.self) {
            try await archive.trash(id: value.id, expectedRevision: nil)
        }
        value.turns[0].start = -1
        let bytes = try JSONEncoder().encode(value)
        try bytes.write(to: folder.appending(path: "broken.json"), options: .atomic)
        let loaded = try await archive.load(id: value.id)
        guard case .readOnly = loaded else {
            Issue.record("Expected read-only history")
            return
        }
        let revision = try await archive.list().first?.revision
        let trashed = try await archive.trash(id: value.id, expectedRevision: revision)
        defer { try? FileManager.default.removeItem(at: trashed) }
        #expect(try Data(contentsOf: trashed.appending(path: "broken.json")) == bytes)
        #expect(try await archive.list().isEmpty)
    }

    @Test("A superseded first save cannot change selection after New")
    func selectedSaveOrdering() async throws {
        let folder = directory()
        defer { try? FileManager.default.removeItem(at: folder) }
        let archive = ConversationArchive(directory: folder)
        let writer = UUID()
        let first = snapshot("first", at: 100)
        var newer = first
        newer.revision += 1
        try await archive.save(newer, expectedRevision: 0, writer: writer, selectAfterSave: true)
        try await archive.select(id: nil)
        await #expect(throws: ConversationArchive.Failure.self) {
            try await archive.save(first, expectedRevision: 0, writer: writer, selectAfterSave: true)
        }
        #expect(try await archive.latest() == nil)
    }

    private func directory() -> URL {
        URL.temporaryDirectory.appending(path: "conversation-library-" + UUID().uuidString)
    }

    private func snapshot(_ id: String, at seconds: Double) -> ConversationSnapshot {
        let date = Date(timeIntervalSince1970: seconds)
        var state = ConversationState(id: id, now: date)
        _ = ConversationFeature.reduce(&state, .send(id, mode: .chat, now: date))
        _ = ConversationFeature.reduce(&state, .generationEnded(run: 1, stopReason: "stopToken"))
        return ConversationSnapshot(state)
    }
}
