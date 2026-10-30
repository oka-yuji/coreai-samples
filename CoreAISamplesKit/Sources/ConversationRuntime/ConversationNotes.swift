import ConversationFeature
import CryptoKit
import Foundation

public struct NoteReceipt: Equatable, Sendable {
    public var id: String
    public var title: String
    public var fileName: String
    public var path: URL
    public var content: String
    public var parts: [String]
    public var part: Int
    public var duplicate: Bool
    public var draft: Bool
    public var warning: String?
}

public struct NoteMoveResult: Equatable, Sendable {
    public var note: NoteReceipt
    public var warning: String?
}

public enum NoteStorageError: Error, LocalizedError {
    case invalidID, invalidRecord, changedFile, moved, missing, invalidDestination, awaitingChecks, staleCheck

    public var errorDescription: String? {
        switch self {
        case .invalidID: "The note_id is invalid. Use the identifier returned by write_note."
        case .invalidRecord: "The saved note record is invalid. Start a new note."
        case .changedFile: "The note was changed or removed outside the app. Its existing content has been preserved."
        case .moved: "The note was moved. Start a new note to add more content."
        case .missing: "This note_id was not found in the current notes folder."
        case .invalidDestination: "The destination is outside the allowed folders or already contains this file."
        case .awaitingChecks: "The note is still a draft. Finish the task before moving it."
        case .staleCheck: "The note changed after it was checked. Its draft has been preserved."
        }
    }
}

public actor ConversationNotes {
    private struct Record: Codable {
        var formatVersion = 1
        var id: String
        var title: String
        var fileName: String
        var parts: NoteParts
        var publishedHash: String?
        var movedTo: URL?
    }

    private struct Located {
        var record: Record
        var folder: URL
        var root: URL
        var draft: Bool
    }

    public init() {}

    public func write(title: String, fileName: String, content: String, directory: URL, stableID: Bool = false) throws
        -> NoteReceipt
    {
        try Task.checkCancellation()
        let parts = try NoteParts(first: content)
        try validName(fileName)
        let root = directory.standardizedFileURL.resolvingSymlinksInPath()
        let drafts = try child(".draft", in: root)
        try FileManager.default.createDirectory(at: drafts, withIntermediateDirectories: true)
        let name = try freeName(fileName, root: root)
        let id = stableID ? deterministicID(name + "\n" + title + "\n" + content) : UUID().uuidString.lowercased()
        let record = Record(id: id, title: title, fileName: name, parts: parts)
        let temporary = try child(".pending-" + id, in: drafts)
        let folder = try child(id, in: drafts)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false)
        do {
            try store(record, in: temporary)
            try FileManager.default.moveItem(at: temporary, to: folder)
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
        return receipt(Located(record: record, folder: folder, root: root, draft: true), duplicate: false)
    }

    public func append(noteID: String, part: Int, content: String, directory: URL) throws -> NoteReceipt {
        try Task.checkCancellation()
        var located = try locate(noteID, directory: directory)
        var record = located.record
        let changed = try record.parts.append(part: part, content: content)
        if record.movedTo != nil {
            guard !changed else { throw NoteStorageError.moved }
            return receipt(located, duplicate: true)
        }
        try verifyPublished(located.record, root: located.root)
        if !located.draft { record.publishedHash = hash(markdown(located.record)) }
        if changed, !located.draft {
            let drafts = try child(".draft", in: located.root)
            try FileManager.default.createDirectory(at: drafts, withIntermediateDirectories: true)
            let target = try child(record.id, in: drafts)
            if located.folder != target { try FileManager.default.moveItem(at: located.folder, to: target) }
            located.folder = target
            located.draft = true
        }
        try storeRecord(record, in: located.folder)
        located.record = record
        var note = receipt(located, duplicate: !changed)
        do { try markdown(record).write(to: child(record.fileName, in: located.folder), options: .atomic) } catch {
            note.warning =
                "The note part was saved, but its Markdown export needs a retry: " + error.localizedDescription
        }
        return note
    }

    public func read(noteID: String, directory: URL) throws -> NoteReceipt {
        receipt(try locate(noteID, directory: directory), duplicate: false)
    }

    public func publish(_ checked: NoteReceipt, directory: URL) throws -> NoteReceipt {
        try Task.checkCancellation()
        var located = try locate(checked.id, directory: directory)
        guard located.record.parts.parts == checked.parts, located.record.title == checked.title,
            located.record.fileName == checked.fileName
        else { throw NoteStorageError.staleCheck }
        guard located.record.movedTo == nil else { return receipt(located, duplicate: false) }
        try verifyPublished(located.record, root: located.root)
        let target = try child(located.record.fileName, in: located.root)
        let bytes = markdown(located.record)
        try bytes.write(to: child(located.record.fileName, in: located.folder), options: .atomic)
        let records = try child(".notes", in: located.root)
        try FileManager.default.createDirectory(at: records, withIntermediateDirectories: true)
        let final = try child(located.record.id, in: records)
        if located.folder != final, FileManager.default.fileExists(atPath: final.path) {
            throw NoteStorageError.invalidRecord
        }
        try bytes.write(to: target, options: .atomic)
        located.record.publishedHash = hash(bytes)
        located.draft = false
        var warning: String?
        do {
            try storeRecord(located.record, in: located.folder)
            if located.folder != final {
                try FileManager.default.moveItem(at: located.folder, to: final)
                located.folder = final
            }
        } catch {
            warning = "The note was published, but its saved record needs a retry: " + error.localizedDescription
        }
        var note = receipt(located, duplicate: false)
        note.warning = warning
        return note
    }

    public func hold(noteIDs: [String], reason: String, directory: URL) throws {
        try Task.checkCancellation()
        let held = try noteIDs.map { try locate($0, directory: directory) }.filter(\.draft)
        guard let first = held.first else { return }
        let lines = ["reason: " + reason, "drafts:"] + held.map { "- " + $0.record.id + "/" + $0.record.fileName }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(
            to: child("HELD.txt", in: first.folder.deletingLastPathComponent()), options: .atomic)
    }

    public func matching(fileNames: [String], directory: URL) throws -> NoteReceipt? {
        let root = directory.standardizedFileURL.resolvingSymlinksInPath()
        for folder in try recordFolders(root) {
            guard let located = try? locate(folder.lastPathComponent, directory: root) else { continue }
            if fileNames.contains(located.record.fileName), located.record.movedTo == nil {
                return receipt(located, duplicate: false)
            }
        }
        return nil
    }

    public func moved(noteID: String, to destination: URL, directory: URL) throws {
        try Task.checkCancellation()
        var located = try locate(noteID, directory: directory)
        located.record.movedTo = destination
        try store(located.record, in: located.folder)
    }

    public func move(
        noteID: String, destination: URL, allowedRoots: [URL], directory: URL
    ) throws -> NoteMoveResult {
        try Task.checkCancellation()
        let located = try locate(noteID, directory: directory)
        if located.draft { throw NoteStorageError.awaitingChecks }
        let destination = destination.standardizedFileURL.resolvingSymlinksInPath()
        guard FolderIdentity.isWithin(destination, anyOf: allowedRoots) else {
            throw NoteStorageError.invalidDestination
        }
        let target = try child(located.record.fileName, in: destination)
        guard !FileManager.default.fileExists(atPath: target.path) else { throw NoteStorageError.invalidDestination }
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        try verifyPublished(located.record, root: located.root)
        var note = receipt(located, duplicate: false)
        do {
            try FileManager.default.moveItem(at: note.path, to: target)
        } catch {
            return NoteMoveResult(
                note: note, warning: "The note is saved, but moving it failed: " + error.localizedDescription)
        }
        note.path = target
        do {
            var moved = try locate(noteID, directory: directory)
            moved.record.movedTo = target
            try store(moved.record, in: moved.folder)
        } catch {
            return NoteMoveResult(
                note: note,
                warning: "The file was moved, but its note record could not be updated. Do not repeat the move: "
                    + error.localizedDescription)
        }
        return NoteMoveResult(note: note, warning: nil)
    }

    private func locate(_ id: String, directory: URL) throws -> Located {
        guard let uuid = UUID(uuidString: id), uuid.uuidString.lowercased() == id.lowercased() else {
            throw NoteStorageError.invalidID
        }
        let id = id.lowercased()
        let root = directory.standardizedFileURL.resolvingSymlinksInPath()
        for name in [".draft", ".notes"] {
            let parent = try child(name, in: root)
            let folder = try child(id, in: parent)
            let file = try child(".record.json", in: folder)
            guard FileManager.default.fileExists(atPath: file.path) else { continue }
            let record = try JSONDecoder().decode(Record.self, from: Data(contentsOf: file))
            guard record.id == id, record.formatVersion == 1 else { throw NoteStorageError.invalidRecord }
            try validName(record.fileName)
            try record.parts.validate()
            let target = try child(record.fileName, in: root)
            let published = (try? Data(contentsOf: target)).map { $0 == markdown(record) } == true
            return Located(
                record: record, folder: folder, root: root,
                draft: name == ".draft" && !published && record.movedTo == nil)
        }
        throw NoteStorageError.missing
    }

    private func store(_ record: Record, in folder: URL) throws {
        try storeRecord(record, in: folder)
        try markdown(record).write(to: child(record.fileName, in: folder), options: .atomic)
    }

    private func storeRecord(_ record: Record, in folder: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(record).write(to: child(".record.json", in: folder), options: .atomic)
    }

    private func markdown(_ record: Record) -> Data {
        let content = record.parts.content
        let title = record.title.trimmingCharacters(in: .whitespacesAndNewlines)
        let heading =
            title.isEmpty || content.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("#")
            ? "" : "# " + title + "\n\n"
        return Data((heading + content + "\n").utf8)
    }

    private func verifyPublished(_ record: Record, root: URL) throws {
        let file = try child(record.fileName, in: root)
        guard FileManager.default.fileExists(atPath: file.path) else {
            if record.publishedHash != nil { throw NoteStorageError.changedFile }
            return
        }
        let actual = hash(try Data(contentsOf: file))
        guard actual == record.publishedHash || actual == hash(markdown(record)) else {
            throw NoteStorageError.changedFile
        }
    }

    private func receipt(_ located: Located, duplicate: Bool) -> NoteReceipt {
        let record = located.record
        let path = record.movedTo ?? (located.draft ? located.folder : located.root).appending(path: record.fileName)
        return NoteReceipt(
            id: record.id, title: record.title, fileName: record.fileName, path: path,
            content: record.parts.content, parts: record.parts.parts, part: record.parts.parts.count,
            duplicate: duplicate, draft: located.draft)
    }

    private func freeName(_ desired: String, root: URL) throws -> String {
        let folders = try recordFolders(root)
        let names = try folders.flatMap { folder in
            try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "md" }.map(\.lastPathComponent)
        }
        for number in 1...99 {
            let name = number == 1 ? desired : String(desired.dropLast(3)) + " \(number).md"
            if !names.contains(where: {
                $0.compare(name, options: [.caseInsensitive, .diacriticInsensitive]) == .orderedSame
            }),
                !FileManager.default.fileExists(atPath: root.appending(path: name).path)
            {
                return name
            }
        }
        throw NoteStorageError.changedFile
    }

    private func recordFolders(_ root: URL) throws -> [URL] {
        var folders: [URL] = []
        for name in [".draft", ".notes"] {
            let parent = try child(name, in: root)
            guard FileManager.default.fileExists(atPath: parent.path) else { continue }
            let entries = try FileManager.default.contentsOfDirectory(
                at: parent,
                includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            for entry in entries where UUID(uuidString: entry.lastPathComponent) != nil {
                let values = try entry.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
                if values.isDirectory == true, values.isSymbolicLink != true {
                    folders.append(try child(entry.lastPathComponent, in: parent))
                }
            }
        }
        return folders
    }

    private func validName(_ name: String) throws {
        guard !name.hasPrefix("."), (name as NSString).lastPathComponent == name, name.hasSuffix(".md") else {
            throw NoteStorageError.invalidRecord
        }
    }

    private func child(_ name: String, in parent: URL) throws -> URL {
        guard parent.isFileURL, !name.isEmpty, name != ".", name != "..",
            (name as NSString).lastPathComponent == name
        else { throw NoteStorageError.invalidRecord }
        let result = parent.appendingPathComponent(name)
        if (try? FileManager.default.destinationOfSymbolicLink(atPath: result.path)) != nil {
            throw NoteStorageError.invalidRecord
        }
        return result
    }

    private func deterministicID(_ text: String) -> String {
        let value = hash(Data(text.utf8))
        return
            "\(value.prefix(8))-\(value.dropFirst(8).prefix(4))-\(value.dropFirst(12).prefix(4))-\(value.dropFirst(16).prefix(4))-\(value.dropFirst(20).prefix(12))"
    }

    private func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

public enum FolderIdentity {
    private struct Key: Hashable {
        var device: dev_t
        var inode: ino_t
    }

    public static func isWithin(_ url: URL, anyOf roots: [URL]) -> Bool {
        let allowed = Set(
            roots.filter(\.isFileURL).compactMap { key($0.standardizedFileURL.resolvingSymlinksInPath().path) })
        guard url.isFileURL, !allowed.isEmpty else { return false }
        var path = url.standardizedFileURL.resolvingSymlinksInPath().path
        var entry = stat()
        while lstat(path, &entry) != 0 {
            let parent = (path as NSString).deletingLastPathComponent
            guard parent != path else { return false }
            path = parent
        }
        var visited = Set<Key>()
        while let current = key(path), visited.insert(current).inserted {
            if allowed.contains(current) { return true }
            path += "/.."
        }
        return false
    }

    public static func isSame(_ url: URL, as other: URL) -> Bool {
        guard url.isFileURL, other.isFileURL, let first = key(url.path) else { return false }
        return first == key(other.path)
    }

    private static func key(_ path: String) -> Key? {
        var info = stat()
        guard stat(path, &info) == 0 else { return nil }
        return Key(device: info.st_dev, inode: info.st_ino)
    }
}
