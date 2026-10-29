import Foundation
import Markdown
import RAGContext

extension RAGStructureReader {
    static func markdown(_ source: RAGDocumentSnapshot) -> RAGDocumentStructure {
        guard source.raw.utf8.elementsEqual(source.text.utf8) else { return .unavailable("normalizationMismatch") }
        let document = Document(parsing: source.raw)
        let positions = RAGTextCoordinates(source.text)
        var headings: [RAGHeading] = []
        for child in document.children {
            guard let heading = child as? Heading else { continue }
            guard let range = heading.range,
                let start = positions.characterOffset(line: range.lowerBound.line, column: range.lowerBound.column),
                let end = headingEnd(heading, source: source.text, positions: positions),
                let boundary = positions.characterOffset(line: range.lowerBound.line, column: 1)
            else { return .unavailable("unmappedMarkdownRange") }
            headings.append(
                RAGHeading(
                    level: heading.level, title: heading.plainText,
                    range: RAGTextRange(start, end), boundaryStart: boundary))
            if headings.count > 256 { return .unavailable("headingLimit") }
        }
        return .make(documentID: source.id, text: source.text, headings: headings)
    }

    private static func headingEnd(_ heading: Heading, source: String, positions: RAGTextCoordinates) -> Int? {
        guard let range = heading.range, let first = positions.lineRange(range.lowerBound.line),
            let firstText = first.text(in: source)
        else { return nil }
        let trimmed = firstText.drop(while: { $0 == " " })
        let hashes = trimmed.prefix(while: { $0 == "#" }).count
        let rest = trimmed.dropFirst(hashes)
        if hashes == heading.level && (rest.isEmpty || rest.first?.isWhitespace == true) {
            return first.upperBound
        }
        guard heading.level <= 2, range.upperBound.line > range.lowerBound.line else { return nil }
        let marker: Character = heading.level == 1 ? "=" : "-"
        for line in (range.lowerBound.line + 1)...range.upperBound.line {
            guard let candidate = positions.lineRange(line), let text = candidate.text(in: source) else { return nil }
            let start = text.drop(while: { $0 == " " })
            guard !start.isEmpty else { return nil }
            guard text.count - start.count <= 3 else { continue }
            let body = start.trimmingCharacters(in: .whitespaces)
            if !body.isEmpty, body.allSatisfy({ $0 == marker }) { return candidate.upperBound }
        }
        return nil
    }
}
