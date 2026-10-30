import ConversationFeature
import ConversationRuntime
import Foundation
import FoundationModels
import Testing

struct ConversationNotesTests {
    @Test("Note parts survive restart, append in order, and identical retries do not duplicate content")
    func durableParts() async throws {
        let directory = folder()
        defer { try? FileManager.default.removeItem(at: directory) }
        let notes = ConversationNotes()
        let first = try await notes.write(
            title: "Plan", fileName: "Plan.md", content: "First part", directory: directory)
        #expect(first.part == 1 && first.draft && first.path.path.contains("/.draft/"))
        let reopened = ConversationNotes()
        #expect(try await reopened.read(noteID: first.id, directory: directory).content == "First part")
        let second = try await reopened.append(noteID: first.id, part: 2, content: "Second part", directory: directory)
        let retry = try await reopened.append(noteID: first.id, part: 2, content: "Second part", directory: directory)
        #expect(second.content == "First part\n\nSecond part" && retry.content == second.content && retry.duplicate)
        #expect(try String(contentsOf: retry.path, encoding: .utf8) == "# Plan\n\nFirst part\n\nSecond part\n")
        #expect(!FileManager.default.fileExists(atPath: directory.appending(path: "Plan.md").path))
        let published = try await reopened.publish(second, directory: directory)
        #expect(!published.draft && published.path == directory.appending(path: "Plan.md"))
        let third = try await reopened.append(noteID: first.id, part: 3, content: "Third part", directory: directory)
        #expect(third.draft && third.content.hasSuffix("Third part"))
        #expect(try String(contentsOf: published.path, encoding: .utf8) == "# Plan\n\nFirst part\n\nSecond part\n")
        let final = try await reopened.publish(third, directory: directory)
        #expect(try String(contentsOf: final.path, encoding: .utf8).hasSuffix("Third part\n"))
    }

    @Test("Missing, changed, oversized and excess parts are rejected without changing the saved draft")
    func rejectsInvalidParts() async throws {
        let directory = folder()
        defer { try? FileManager.default.removeItem(at: directory) }
        let notes = ConversationNotes()
        let first = try await notes.write(title: "Plan", fileName: "Plan.md", content: "First", directory: directory)
        let before = try Data(contentsOf: first.path)
        await #expect(throws: NotePartError.missing(expected: 2)) {
            try await notes.append(noteID: first.id, part: 3, content: "Gap", directory: directory)
        }
        await #expect(throws: NotePartError.changed(part: 1)) {
            try await notes.append(noteID: first.id, part: 1, content: "Changed", directory: directory)
        }
        await #expect(throws: NotePartError.tooLong) {
            try await notes.append(
                noteID: first.id, part: 2, content: String(repeating: "x", count: 2_001), directory: directory)
        }
        #expect(try Data(contentsOf: first.path) == before)
        for part in 2...4 {
            _ = try await notes.append(noteID: first.id, part: part, content: "Part \(part)", directory: directory)
        }
        let four = try Data(contentsOf: first.path)
        await #expect(throws: NotePartError.tooMany) {
            try await notes.append(noteID: first.id, part: 5, content: "Fifth", directory: directory)
        }
        #expect(try Data(contentsOf: first.path) == four)
    }

    @Test("Appending preserves manual edits and a moved note cannot be extended at an arbitrary path")
    func protectsExternalChanges() async throws {
        let directory = folder()
        defer { try? FileManager.default.removeItem(at: directory) }
        let notes = ConversationNotes()
        let first = try await notes.write(title: "Plan", fileName: "Plan.md", content: "First", directory: directory)
        let published = try await notes.publish(first, directory: directory)
        try Data("Manual edit".utf8).write(to: published.path, options: .atomic)
        await #expect(throws: NoteStorageError.self) {
            try await notes.append(noteID: first.id, part: 2, content: "Second", directory: directory)
        }
        #expect(try String(contentsOf: published.path, encoding: .utf8) == "Manual edit")
        let moved = directory.appending(path: "Moved.md")
        try FileManager.default.moveItem(at: published.path, to: moved)
        try await notes.moved(noteID: first.id, to: moved, directory: directory)
        await #expect(throws: NoteStorageError.self) {
            try await notes.append(noteID: first.id, part: 2, content: "Second", directory: directory)
        }
        #expect(try await notes.append(noteID: first.id, part: 1, content: "First", directory: directory).duplicate)
        #expect(try String(contentsOf: moved, encoding: .utf8) == "Manual edit")
    }

    @Test("The append tool uses note_id on the wire while Swift uses noteID")
    func schemaNames() throws {
        let arguments = try AppendNoteArguments(
            GeneratedContent(json: #"{"note_id":"note-1","part":2,"content":"Body"}"#))
        #expect(arguments.noteID == "note-1" && arguments.part == 2 && arguments.content == "Body")
        #expect(try arguments.generatedContent.value(String.self, forProperty: "note_id") == "note-1")
        let schema = try JSONSerialization.jsonObject(with: JSONEncoder().encode(AppendNoteArguments.generationSchema))
        let text = String(describing: schema)
        #expect(text.contains("note_id") && !text.contains("noteID"))
    }

    @Test("Stable validation runs return the same note identifier in independent folders")
    func stableIdentifiers() async throws {
        let firstFolder = folder()
        let secondFolder = folder()
        defer {
            try? FileManager.default.removeItem(at: firstFolder)
            try? FileManager.default.removeItem(at: secondFolder)
        }
        let notes = ConversationNotes()
        let first = try await notes.write(
            title: "Stable", fileName: "Stable.md", content: "Body", directory: firstFolder, stableID: true)
        let second = try await notes.write(
            title: "Stable", fileName: "Stable.md", content: "Body", directory: secondFolder, stableID: true)
        #expect(first.id == second.id && first.path != second.path)
        #expect(UUID(uuidString: first.id) != nil)
    }

    @Test("Cancellation before a note operation begins creates no files")
    func cancelledWrite() async {
        let directory = folder()
        defer { try? FileManager.default.removeItem(at: directory) }
        let notes = ConversationNotes()
        await withTaskGroup(of: Bool.self) { group in
            group.cancelAll()
            group.addTask {
                do {
                    _ = try await notes.write(
                        title: "Cancelled", fileName: "Cancelled.md", content: "Body", directory: directory)
                    return false
                } catch is CancellationError { return true } catch { return false }
            }
            #expect(await group.next() == true)
        }
        #expect(!FileManager.default.fileExists(atPath: directory.path))
    }

    @Test("Moving a managed note respects the destination and draft guard and keeps its identifier")
    func movesManagedNote() async throws {
        let root = folder()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appending(path: "notes")
        let target = root.appending(path: "archive")
        let notes = ConversationNotes()
        let note = try await notes.write(title: "Plan", fileName: "Plan.md", content: "Body", directory: directory)
        await #expect(throws: NoteStorageError.self) {
            try await notes.move(
                noteID: note.id, destination: target, allowedRoots: [root], directory: directory)
        }
        #expect(try await notes.read(noteID: note.id, directory: directory).draft)
        try Data(contentsOf: note.path).write(to: directory.appending(path: "Plan.md"), options: .atomic)
        let moved = try await notes.move(
            noteID: note.id, destination: target, allowedRoots: [root], directory: directory)
        #expect(moved.warning == nil && moved.note.id == note.id && !moved.note.draft)
        #expect(try await notes.read(noteID: note.id, directory: directory).draft == false)
        #expect(FileManager.default.fileExists(atPath: target.appending(path: "Plan.md").path))
        #expect(!FileManager.default.fileExists(atPath: directory.appending(path: "Plan.md").path))
        await #expect(throws: NoteStorageError.self) {
            try await notes.append(noteID: note.id, part: 2, content: "Too late", directory: directory)
        }
    }

    @Test(
        "On a case-sensitive volume a move into a separate folder spelled like the only allowed folder is refused and the note stays",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_CASE_SENSITIVE_ROOT"] != nil))
    func refusesCaseVariantFolderOnCaseSensitiveVolume() async throws {
        let volume = try #require(ProcessInfo.processInfo.environment["COREAI_CASE_SENSITIVE_ROOT"])
        let root = URL(fileURLWithPath: volume, isDirectory: true).appending(path: "notes-case-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let allowed = root.appending(path: "Allowed")
        let other = root.appending(path: "allowed")
        try FileManager.default.createDirectory(at: allowed, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        let volumeValues = try root.resourceValues(forKeys: [.volumeSupportsCaseSensitiveNamesKey])
        try #require(volumeValues.volumeSupportsCaseSensitiveNames == true)
        try #require(FileManager.default.contentsOfDirectory(atPath: root.path).sorted() == ["Allowed", "allowed"])
        let directory = root.appending(path: "notes")
        let notes = ConversationNotes()
        let draft = try await notes.write(title: "Plan", fileName: "Plan.md", content: "Body", directory: directory)
        let note = try await notes.publish(draft, directory: directory)
        await #expect(throws: NoteStorageError.invalidDestination) {
            try await notes.move(noteID: note.id, destination: other, allowedRoots: [allowed], directory: directory)
        }
        #expect(try await notes.read(noteID: note.id, directory: directory).path == note.path)
        #expect(FileManager.default.fileExists(atPath: note.path.path))
        #expect(try FileManager.default.contentsOfDirectory(atPath: other.path).isEmpty)
        #expect(try FileManager.default.contentsOfDirectory(atPath: allowed.path).isEmpty)
    }

    @Test("On a case-insensitive volume a move into the allowed folder spelled with different case still succeeds")
    func movesIntoCaseVariantOfAllowedFolder() async throws {
        let root = folder()
        defer { try? FileManager.default.removeItem(at: root) }
        let allowed = root.appending(path: "Allowed")
        try FileManager.default.createDirectory(at: allowed, withIntermediateDirectories: true)
        let volumeValues = try root.resourceValues(forKeys: [.volumeSupportsCaseSensitiveNamesKey])
        try #require(volumeValues.volumeSupportsCaseSensitiveNames == false)
        let directory = root.appending(path: "notes")
        let notes = ConversationNotes()
        let draft = try await notes.write(title: "Plan", fileName: "Plan.md", content: "Body", directory: directory)
        let note = try await notes.publish(draft, directory: directory)
        let spelled = root.appending(path: "allowed")
        let moved = try await notes.move(
            noteID: note.id, destination: spelled, allowedRoots: [allowed], directory: directory)
        #expect(moved.warning == nil && !moved.note.draft)
        #expect(try FileManager.default.contentsOfDirectory(atPath: allowed.path) == ["Plan.md"])
        #expect(!FileManager.default.fileExists(atPath: note.path.path))
    }

    @Test(
        "A destination that leaves the allowed folder through a symlink is refused and nothing is written outside",
        arguments: ["link", "link/new", "dangling/new"])
    func refusesSymlinkedDestination(_ relativePath: String) async throws {
        let root = folder()
        defer { try? FileManager.default.removeItem(at: root) }
        let allowed = root.appending(path: "Allowed")
        let outside = root.appending(path: "Outside")
        try FileManager.default.createDirectory(at: allowed, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: allowed.appending(path: "link"), withDestinationURL: outside)
        try FileManager.default.createSymbolicLink(
            at: allowed.appending(path: "dangling"), withDestinationURL: outside.appending(path: "missing"))
        let directory = root.appending(path: "notes")
        let notes = ConversationNotes()
        let draft = try await notes.write(title: "Plan", fileName: "Plan.md", content: "Body", directory: directory)
        let note = try await notes.publish(draft, directory: directory)
        await #expect(throws: NoteStorageError.invalidDestination) {
            try await notes.move(
                noteID: note.id, destination: allowed.appending(path: relativePath), allowedRoots: [allowed],
                directory: directory)
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: outside.path).isEmpty)
        #expect(FileManager.default.fileExists(atPath: note.path.path))
    }

    @Test("The same-folder check compares the folder itself rather than its path")
    func sameFolder() throws {
        let root = folder()
        defer { try? FileManager.default.removeItem(at: root) }
        let notes = root.appending(path: "Notes")
        let other = root.appending(path: "Other")
        try FileManager.default.createDirectory(at: notes, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: root.appending(path: "alias"), withDestinationURL: notes)
        #expect(FolderIdentity.isSame(root.appending(path: "alias"), as: notes))
        #expect(!FolderIdentity.isSame(other, as: notes))
        #expect(!FolderIdentity.isSame(root.appending(path: "missing"), as: notes))
    }

    private func folder() -> URL { URL.temporaryDirectory.appending(path: "conversation-notes-" + UUID().uuidString) }

    @Test("Draft file names reserve case-insensitive names and private tmp paths are accepted")
    func canonicalPathsAndNames() async throws {
        let directory = URL(fileURLWithPath: "/private/tmp/notes-path-test-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let notes = ConversationNotes()
        let first = try await notes.write(title: "Plan", fileName: "Plan.md", content: "First", directory: directory)
        let second = try await notes.write(title: "plan", fileName: "plan.md", content: "Second", directory: directory)
        #expect(first.fileName == "Plan.md" && second.fileName == "plan 2.md")
        _ = try await notes.publish(first, directory: directory)
        _ = try await notes.publish(second, directory: directory)
    }

    @Test("A published note whose metadata update was interrupted can be appended without losing its prior version")
    func resumesPublishedMetadataFailure() async throws {
        let directory = folder()
        defer { try? FileManager.default.removeItem(at: directory) }
        let notes = ConversationNotes()
        let first = try await notes.write(title: "Plan", fileName: "Plan.md", content: "First", directory: directory)
        try Data(contentsOf: first.path).write(to: directory.appending(path: "Plan.md"), options: .atomic)
        #expect(try await notes.read(noteID: first.id, directory: directory).draft == false)
        let second = try await notes.append(noteID: first.id, part: 2, content: "Second", directory: directory)
        #expect(second.draft)
        let published = try await notes.publish(second, directory: directory)
        #expect(try String(contentsOf: published.path, encoding: .utf8).contains("First\n\nSecond"))
    }

    @Test("A redirected draft directory is refused without writing into the external folder")
    func refusesSymlinkDrafts() async throws {
        let directory = folder()
        let external = folder()
        defer {
            try? FileManager.default.removeItem(at: directory)
            try? FileManager.default.removeItem(at: external)
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: directory.appending(path: ".draft"), withDestinationURL: external)
        await #expect(throws: NoteStorageError.self) {
            try await ConversationNotes().write(
                title: "Plan", fileName: "Plan.md", content: "Body", directory: directory)
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: external.path).isEmpty)
    }

    @Test("Publishing refuses parts added after the checked receipt and preserves the draft")
    func refusesStaleCheck() async throws {
        let directory = folder()
        defer { try? FileManager.default.removeItem(at: directory) }
        let notes = ConversationNotes()
        let first = try await notes.write(title: "Plan", fileName: "Plan.md", content: "First", directory: directory)
        let second = try await notes.append(noteID: first.id, part: 2, content: "Second", directory: directory)
        await #expect(throws: NoteStorageError.self) { try await notes.publish(first, directory: directory) }
        #expect(!FileManager.default.fileExists(atPath: directory.appending(path: "Plan.md").path))
        #expect(try await notes.read(noteID: first.id, directory: directory).parts == second.parts)
        _ = try await notes.publish(second, directory: directory)
    }

    @Test("A failed Markdown export returns the saved parts and can be retried without publishing unchecked content")
    func repairsFailedExport() async throws {
        let directory = folder()
        defer { try? FileManager.default.removeItem(at: directory) }
        let notes = ConversationNotes()
        let first = try await notes.write(title: "Plan", fileName: "Plan.md", content: "First", directory: directory)
        try FileManager.default.removeItem(at: first.path)
        try FileManager.default.createDirectory(at: first.path, withIntermediateDirectories: false)
        let second = try await notes.append(noteID: first.id, part: 2, content: "Second", directory: directory)
        #expect(second.warning != nil && second.parts == ["First", "Second"] && second.draft)
        await #expect(throws: (any Error).self) { try await notes.publish(second, directory: directory) }
        #expect(!FileManager.default.fileExists(atPath: directory.appending(path: "Plan.md").path))
        try FileManager.default.removeItem(at: first.path)
        let retry = try await notes.append(noteID: first.id, part: 2, content: "Second", directory: directory)
        #expect(retry.duplicate && retry.warning == nil)
        let published = try await notes.publish(retry, directory: directory)
        #expect(try String(contentsOf: published.path, encoding: .utf8) == "# Plan\n\nFirst\n\nSecond\n")
    }
}
