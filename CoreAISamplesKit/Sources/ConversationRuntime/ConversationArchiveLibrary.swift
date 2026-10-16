import ConversationFeature
import Foundation

public struct ConversationRenameResult: Equatable, Sendable {
    public var snapshot: ConversationSnapshot
    public var exportError: String?
}

extension ConversationArchive {
    private struct RevisionHeader: Decodable { var revision: Int }

    private struct SummaryHeader: Decodable {
        struct Turn: Decodable { var mode: ConversationMode }
        var id: String
        var title: String
        var updatedAt: Date
        var formatVersion: Int
        var revision: Int?
        var turns: [Turn]
    }

    public func list() throws -> [ConversationSummary] {
        guard FileManager.default.fileExists(atPath: directory.path) else { return [] }
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.contentModificationDateKey])
        var items: [ConversationSummary] = []
        for file in files where file.pathExtension == "json" {
            let id = file.deletingPathExtension().lastPathComponent
            let safe = try path(id, extension: "json")
            let data = try Data(contentsOf: safe)
            if let header = try? JSONDecoder().decode(SummaryHeader.self, from: data), header.id == id {
                items.append(
                    ConversationSummary(
                        id: id, title: header.title, updatedAt: header.updatedAt,
                        mode: header.turns.last?.mode,
                        revision: header.formatVersion == ConversationSnapshot.currentFormatVersion
                            ? header.revision : nil))
            } else {
                let date =
                    try file.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
                    ?? .distantPast
                items.append(ConversationSummary(id: id, title: id, updatedAt: date, mode: nil, revision: nil))
            }
        }
        return items.sorted { $0.updatedAt == $1.updatedAt ? $0.id < $1.id : $0.updatedAt > $1.updatedAt }
    }

    public func select(id: String?) throws {
        if let id { _ = try load(id: id) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONEncoder().encode(id).write(to: directory.appending(path: ".selection"), options: .atomic)
    }

    public func rename(id: String, title: String, expectedRevision: Int, now: Date, writer: UUID) throws
        -> ConversationRenameResult
    {
        let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty, case .snapshot(var snapshot) = try load(id: id) else { throw Failure.invalidSnapshot }
        guard snapshot.revision == expectedRevision else { throw Failure.conflictingRevision }
        snapshot.title = title
        snapshot.updatedAt = now
        snapshot.revision += 1
        do {
            try save(snapshot, expectedRevision: expectedRevision, writer: writer)
            return ConversationRenameResult(snapshot: snapshot, exportError: nil)
        } catch {
            let json = try path(id, extension: "json")
            if let written = try? JSONDecoder().decode(ConversationSnapshot.self, from: Data(contentsOf: json)),
                written == snapshot
            {
                return ConversationRenameResult(snapshot: snapshot, exportError: error.localizedDescription)
            }
            throw error
        }
    }

    public func repairExport(id: String) throws {
        guard case .snapshot(let snapshot) = try load(id: id) else { throw Failure.invalidSnapshot }
        try Data(snapshot.markdown().utf8).write(to: path(id, extension: "md"), options: .atomic)
    }

    public func trash(id: String, expectedRevision: Int?) throws -> URL {
        try trash(id: id, expectedRevision: expectedRevision, move: Self.moveToTrash)
    }

    func trash(id: String, expectedRevision: Int?, move: @Sendable (URL) throws -> URL) throws -> URL {
        let json = try path(id, extension: "json")
        let markdown = try path(id, extension: "md")
        let archived = try load(id: id)
        if let expectedRevision {
            let header = try JSONDecoder().decode(RevisionHeader.self, from: Data(contentsOf: json))
            guard header.revision == expectedRevision else { throw Failure.conflictingRevision }
        } else if case .snapshot = archived {
            throw Failure.conflictingRevision
        }
        let staging = directory.appending(
            path: "Conversation-" + id + "-" + UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        var moved: [(source: URL, staged: URL)] = []
        do {
            for file in [json, markdown] where FileManager.default.fileExists(atPath: file.path) {
                let target = staging.appending(path: file.lastPathComponent)
                try FileManager.default.moveItem(at: file, to: target)
                moved.append((file, target))
            }
            guard !moved.isEmpty else { throw Failure.invalidSnapshot }
            return try move(staging)
        } catch {
            for item in moved { try FileManager.default.moveItem(at: item.staged, to: item.source) }
            try FileManager.default.removeItem(at: staging)
            throw error
        }
    }

    private static func moveToTrash(_ url: URL) throws -> URL {
        #if os(macOS)
            var destination: NSURL?
            try FileManager.default.trashItem(at: url, resultingItemURL: &destination)
            guard let destination else { throw Failure.invalidSnapshot }
            return destination as URL
        #else
            throw Failure.invalidSnapshot
        #endif
    }
}
