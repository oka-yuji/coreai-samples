import Foundation

public struct SpokenTranscript: Equatable, Sendable {
    public let runs: [SpokenRun]
    private let characters: [Character]
    private let sourceRuns: [ClosedRange<Int>]

    public init(runs: [SpokenRun]) {
        self.runs = runs
        let text = runs.map(\.text).joined()
        var characters: [Character] = []
        var sourceRuns: [ClosedRange<Int>] = []
        var currentRun = 0
        var runEnd = runs.first?.text.utf8.count ?? 0
        var byteOffset = 0

        func owner(ofByte offset: Int) -> Int {
            while runEnd <= offset {
                currentRun += 1
                runEnd += runs[currentRun].text.utf8.count
            }
            return currentRun
        }

        var index = text.startIndex
        while index < text.endIndex {
            let next = text.index(after: index)
            let first = owner(ofByte: byteOffset)
            byteOffset += text.utf8.distance(from: index, to: next)
            characters.append(text[index])
            sourceRuns.append(first...owner(ofByte: byteOffset - 1))
            index = next
        }
        self.characters = characters
        self.sourceRuns = sourceRuns
    }

    public func matches(_ query: String) -> [SpokenHit] {
        let pattern = Array(query.trimmingCharacters(in: .whitespacesAndNewlines))
        guard !pattern.isEmpty, pattern.count <= characters.count else { return [] }
        let prefixLengths = Self.prefixLengths(of: pattern)
        var matched = 0
        var hits: [SpokenHit] = []
        for (offset, character) in characters.enumerated() {
            matched = Self.advance(matched, with: character, in: pattern, prefixes: prefixLengths)
            if matched == pattern.count {
                let end = offset + 1
                if let hit = hit(for: (end - matched)..<end) { hits.append(hit) }
                matched = prefixLengths[matched - 1]
            }
        }
        return hits.sorted { ($0.start, $0.id) < ($1.start, $1.id) }
    }

    private static func prefixLengths(of pattern: [Character]) -> [Int] {
        var lengths = Array(repeating: 0, count: pattern.count)
        for index in pattern.indices.dropFirst() {
            lengths[index] = advance(lengths[index - 1], with: pattern[index], in: pattern, prefixes: lengths)
        }
        return lengths
    }

    private static func advance(
        _ matched: Int, with character: Character, in pattern: [Character], prefixes: [Int]
    ) -> Int {
        var length = matched
        while length > 0 && pattern[length] != character {
            length = prefixes[length - 1]
        }
        return pattern[length] == character ? length + 1 : 0
    }

    private func hit(for match: Range<Int>) -> SpokenHit? {
        guard let first = Self.interval(of: runs[sourceRuns[match.lowerBound].lowerBound]),
            let last = Self.interval(of: runs[sourceRuns[match.upperBound - 1].upperBound]),
            first.lowerBound <= last.upperBound
        else { return nil }
        return SpokenHit(
            id: match.lowerBound, text: excerpt(around: match), start: first.lowerBound, end: last.upperBound)
    }

    private static func interval(of run: SpokenRun) -> ClosedRange<Double>? {
        guard let start = run.start, let end = run.end, start.isFinite, end.isFinite, start >= 0, end >= start else {
            return nil
        }
        return start...end
    }

    private func excerpt(around match: Range<Int>) -> String {
        var lower = match.lowerBound
        let lowerLimit = max(0, lower - 16)
        while lower > lowerLimit && !Self.endsSentence(characters[lower - 1]) {
            lower -= 1
        }
        var upper = min(match.upperBound, lower + 80)
        let upperLimit = min(characters.count, lower + 80, upper + 16)
        while upper < upperLimit && !Self.endsSentence(characters[upper - 1]) {
            upper += 1
        }
        return String(characters[lower..<upper]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func endsSentence(_ character: Character) -> Bool {
        switch character {
        case ".", "!", "?", "\u{3002}", "\u{FF01}", "\u{FF0E}", "\u{FF1F}": true
        default: character.isNewline
        }
    }
}
