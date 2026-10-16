import ConversationFeature
import Foundation

public actor ConversationArchive {
    public enum Failure: Error, LocalizedError {
        case invalidID, invalidSnapshot, conflictingRevision

        public var errorDescription: String? {
            switch self {
            case .invalidID: "The conversation has an invalid file identifier."
            case .invalidSnapshot: "The conversation file is incomplete or unsupported."
            case .conflictingRevision: "A newer or different copy of this conversation is already saved."
            }
        }
    }

    private struct Header: Decodable {
        var formatVersion: Int
        var updatedAt: Date
    }

    public let directory: URL
    private var writers: [String: (id: UUID, revision: Int)] = [:]

    public init(directory: URL) {
        self.directory = directory
    }

    public func save(
        _ snapshot: ConversationSnapshot, expectedRevision: Int, writer: UUID, selectAfterSave: Bool = false
    ) throws {
        try Self.validate(snapshot)
        let json = try path(snapshot.id, extension: "json")
        let markdown = try path(snapshot.id, extension: "md")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: json.path) {
            let existing = try JSONDecoder().decode(ConversationSnapshot.self, from: Data(contentsOf: json))
            let expected = writers[snapshot.id].flatMap { $0.id == writer ? $0.revision : nil } ?? expectedRevision
            guard (existing.revision == expected && existing.revision < snapshot.revision) || existing == snapshot
            else {
                throw Failure.conflictingRevision
            }
        } else if expectedRevision != 0 {
            throw Failure.conflictingRevision
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(snapshot).write(to: json, options: .atomic)
        writers[snapshot.id] = (writer, snapshot.revision)
        try Data(snapshot.markdown().utf8).write(to: markdown, options: .atomic)
        if selectAfterSave { try select(id: snapshot.id) }
    }

    public func load(id: String) throws -> ArchivedConversation {
        let json = try path(id, extension: "json")
        let data = try Data(contentsOf: json)
        do {
            let header = try JSONDecoder().decode(Header.self, from: data)
            guard ConversationSnapshot.supports(version: header.formatVersion) else {
                return try readOnly(id: id, reason: "This conversation uses a newer or unsupported file format.")
            }
            let snapshot = try JSONDecoder().decode(ConversationSnapshot.self, from: data)
            guard snapshot.id == id else { throw Failure.invalidSnapshot }
            try Self.validate(snapshot)
            return .snapshot(snapshot)
        } catch {
            return try readOnly(
                id: id, reason: "This conversation could not be restored: \(error.localizedDescription)")
        }
    }

    public func latest() throws -> ArchivedConversation? {
        guard FileManager.default.fileExists(atPath: directory.path) else { return nil }
        let selection = directory.appending(path: ".selection")
        if FileManager.default.fileExists(atPath: selection.path) {
            let id = try JSONDecoder().decode(String?.self, from: Data(contentsOf: selection))
            guard let id else { return nil }
            if FileManager.default.fileExists(atPath: try path(id, extension: "json").path) { return try load(id: id) }
        }
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.contentModificationDateKey, .isRegularFileKey])
        let candidates = try files.filter { $0.pathExtension == "json" }.map { url in
            let safeURL = try path(url.deletingPathExtension().lastPathComponent, extension: "json")
            let values = try safeURL.resourceValues(forKeys: [.contentModificationDateKey, .isRegularFileKey])
            guard values.isRegularFile == true else { throw Failure.invalidSnapshot }
            let header = try? JSONDecoder().decode(Header.self, from: Data(contentsOf: safeURL))
            return (url: safeURL, date: header?.updatedAt ?? values.contentModificationDate ?? .distantPast)
        }
        guard let newest = candidates.max(by: { $0.date < $1.date }) else { return nil }
        return try load(id: newest.url.deletingPathExtension().lastPathComponent)
    }

    private func readOnly(id: String, reason: String) throws -> ArchivedConversation {
        let markdown = try path(id, extension: "md")
        let text: String
        if FileManager.default.fileExists(atPath: markdown.path) {
            text = try String(contentsOf: markdown, encoding: .utf8)
        } else {
            text = "No Markdown export is available. The original JSON file has been preserved."
        }
        return .readOnly(id: id, markdown: text, reason: reason)
    }

    func path(_ id: String, extension suffix: String) throws -> URL {
        guard !id.isEmpty, id.count <= 128,
            id.unicodeScalars.allSatisfy({
                CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_")).contains($0)
            })
        else { throw Failure.invalidID }
        let url = directory.appending(path: "\(id).\(suffix)")
        if FileManager.default.fileExists(atPath: url.path),
            try url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink == true
        {
            throw Failure.invalidID
        }
        return url
    }

    public static func validate(_ snapshot: ConversationSnapshot) throws {
        guard ConversationSnapshot.supports(version: snapshot.formatVersion), snapshot.revision >= 0,
            snapshot.revision < 1_000_000_000,
            snapshot.createdAt.timeIntervalSince1970.isFinite, snapshot.updatedAt.timeIntervalSince1970.isFinite,
            snapshot.turns.allSatisfy({
                $0.start >= 0 && $0.start <= snapshot.transcript.entries.count && $0.offset >= 0
                    && $0.offset <= 1_000_000
                    && $0.settings.maxSteps > 0 && $0.settings.maxSteps <= 1_000_000
                    && ($0.settings.maxTokens ?? 1) > 0 && ($0.settings.maxTokens ?? 1) <= 1_000_000
            }),
            snapshot.turns.map(\.start) == snapshot.turns.map(\.start).sorted(),
            snapshot.turns.reduce(0, { $0 + $1.steps.count }) <= 1_000_000,
            snapshot.turns.allSatisfy({ $0.steps.allSatisfy(Self.validStats) })
        else { throw Failure.invalidSnapshot }
    }
    private static func validStats(_ stats: StepStats) -> Bool {
        [stats.step, stats.promptTokens, stats.reusedTokens, stats.generatedTokens, stats.toolCalls].allSatisfy {
            (0...1_000_000).contains($0)
        } && [stats.ttftSeconds, stats.tokensPerSecond].compactMap { $0 }.allSatisfy { $0.isFinite && $0 >= 0 }
    }

}
