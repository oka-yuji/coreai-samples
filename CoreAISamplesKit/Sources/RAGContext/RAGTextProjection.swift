import Foundation

public struct RAGTextProjection: Sendable {
    public var text: String
    public var markerEnds: [String: Int]

    public static func removingMarkers(from annotated: String, prefix: String, markers: [String]) -> Self? {
        guard !prefix.isEmpty, Set(markers).count == markers.count else { return nil }
        var output = ""
        var space = false
        var newline = false
        func consume(_ part: Substring) {
            for character in part {
                if character.isWhitespace {
                    if character == "\n" { newline = true } else { space = true }
                    continue
                }
                if !output.isEmpty, newline || space {
                    output.append(newline ? "\n" : " ")
                }
                newline = false
                space = false
                output.append(character)
            }
        }
        var ends: [String: Int] = [:]
        var cursor = annotated.startIndex
        while let start = annotated.range(of: prefix, range: cursor..<annotated.endIndex) {
            consume(annotated[cursor..<start.lowerBound])
            guard let delimiter = annotated[start.upperBound...].firstIndex(of: ";") else { return nil }
            let end = annotated.index(after: delimiter)
            let marker = String(annotated[start.lowerBound..<end])
            guard markers.contains(marker), ends[marker] == nil else { return nil }
            ends[marker] = output.utf8.count
            cursor = end
        }
        consume(annotated[cursor...])
        guard ends.count == markers.count else { return nil }
        let coordinates = RAGTextCoordinates(output)
        var resolved: [String: Int] = [:]
        for (marker, offset) in ends {
            guard let character = coordinates.characterOffset(utf8Offset: offset) else { return nil }
            resolved[marker] = character
        }
        return Self(text: output, markerEnds: resolved)
    }
}
