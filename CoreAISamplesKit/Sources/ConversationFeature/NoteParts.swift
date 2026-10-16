import Foundation

public enum NotePartError: Error, LocalizedError, Equatable {
    case missingReference(String)
    case empty, tooLong, tooMany
    case missing(expected: Int)
    case changed(part: Int)
    case conflictingReference(String)

    public var errorDescription: String? {
        switch self {
        case .empty: "A note part must not be empty."
        case .tooLong: "A note part must contain at most 2000 characters. Split it into smaller parts."
        case .tooMany: "A note can contain at most 4 parts."
        case .missing(let part): "The next note part must be \(part)."
        case .changed(let part): "Part \(part) was already saved with different content. Start a new note to revise it."
        case .missingReference(let reference): "Citation \(reference) has no URL in this part or earlier parts."
        case .conflictingReference(let reference): "Citation \(reference) refers to different sources in this note."
        }
    }
}

public struct NoteParts: Codable, Equatable, Sendable {
    public static let maximumCharacters = 2_000
    public static let maximumParts = 4
    public private(set) var parts: [String]

    public init(first: String) throws {
        parts = []
        _ = try append(part: 1, content: first)
    }

    public var content: String { parts.joined(separator: "\n\n") }

    @discardableResult
    public mutating func append(part: Int, content: String) throws -> Bool {
        guard part > 0 else { throw NotePartError.missing(expected: parts.count + 1) }
        guard part <= Self.maximumParts else { throw NotePartError.tooMany }
        guard !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw NotePartError.empty }
        guard content.count <= Self.maximumCharacters else { throw NotePartError.tooLong }
        if part <= parts.count {
            guard parts[part - 1] == content else { throw NotePartError.changed(part: part) }
            return false
        }
        guard part == parts.count + 1 else { throw NotePartError.missing(expected: parts.count + 1) }
        try Self.checkReferences(parts + [content])
        parts.append(content)
        return true
    }

    public func validate() throws {
        guard let first = parts.first else { throw NotePartError.empty }
        var validated = try NoteParts(first: first)
        for (index, part) in parts.dropFirst().enumerated() { try validated.append(part: index + 2, content: part) }
    }

    private static func checkReferences(_ parts: [String]) throws {
        var references: [String: String] = [:]
        var used: Set<String> = []
        var fence: (marker: Character, count: Int)?
        func register(_ label: String, _ destination: String) throws {
            if let previous = references[label], previous != destination {
                throw NotePartError.conflictingReference("[" + label + "]")
            }
            references[label] = destination
        }
        for line in parts.joined(separator: "\n\n").components(separatedBy: .newlines) {
            let text = line.trimmingCharacters(in: .whitespaces)
            if let marker = text.first, marker == "`" || marker == "~" {
                let count = text.prefix { $0 == marker }.count
                if let active = fence {
                    if active.marker == marker, count >= active.count,
                        text.dropFirst(count).trimmingCharacters(in: .whitespaces).isEmpty
                    {
                        fence = nil
                    }
                } else if count >= 3 {
                    fence = (marker, count)
                    continue
                }
                if count >= 3 { continue }
            }
            guard fence == nil else { continue }
            let visible = text.replacing(/`+[^`]*`+/, with: "")
            if let definition = visible.firstMatch(of: /^\[(\^?[0-9]+)\]:\s*<?(https?:\/\/[^\s>]+)>?/) {
                try register(String(definition.1), String(definition.2))
            }
            for reference in visible.matches(of: /\[(\^?[0-9]+)\]/) {
                let range = reference.range
                if range.lowerBound > visible.startIndex {
                    let before = visible[visible.index(before: range.lowerBound)]
                    if before.isLetter || before.isNumber || before == "_" { continue }
                }
                if range.upperBound < visible.endIndex, [":", "("].contains(visible[range.upperBound]) { continue }
                used.insert(String(reference.1))
            }
            for link in visible.matches(of: /\[([0-9]+)\]\(/) {
                let rest = visible[link.range.upperBound...]
                var destination = ""
                var depth = 0
                for character in rest {
                    if character == ")" {
                        if depth == 0 { break }
                        depth -= 1
                    } else if character == "(" {
                        depth += 1
                    }
                    if character.isWhitespace { break }
                    destination.append(character)
                }
                if destination.lowercased().hasPrefix("https://") || destination.lowercased().hasPrefix("http://") {
                    try register(String(link.1), destination)
                }
            }
        }
        if let missing = used.sorted().first(where: { references[$0] == nil }) {
            throw NotePartError.missingReference("[" + missing + "]")
        }
    }
}
