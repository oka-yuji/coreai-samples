import Foundation

public struct RAGTextCoordinates: Sendable {
    private var offsets: [Int: Int] = [0: 0]
    private var lineStarts = [0]
    private var lineEnds: [Int] = []

    public init(_ text: String) {
        var bytes = 0
        var characters = 0
        for character in text {
            let length = character.utf8.count
            if character == "\n" || character == "\r" || character == "\r\n" {
                lineEnds.append(bytes)
                lineStarts.append(bytes + length)
            }
            bytes += length
            characters += 1
            offsets[bytes] = characters
        }
        lineEnds.append(bytes)
    }

    public func characterOffset(utf8Offset: Int) -> Int? { offsets[utf8Offset] }

    public func lineRange(_ line: Int) -> RAGTextRange? {
        guard line > 0, line <= lineStarts.count,
            let start = offsets[lineStarts[line - 1]], let end = offsets[lineEnds[line - 1]]
        else { return nil }
        return RAGTextRange(start, end)
    }

    public func characterOffset(line: Int, column: Int) -> Int? {
        guard line > 0, line <= lineStarts.count, column > 0 else { return nil }
        let (offset, overflow) = lineStarts[line - 1].addingReportingOverflow(column - 1)
        guard !overflow, offset <= lineEnds[line - 1] else { return nil }
        return offsets[offset]
    }
}
