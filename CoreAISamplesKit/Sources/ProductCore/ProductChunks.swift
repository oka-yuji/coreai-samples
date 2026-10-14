import Foundation

public struct ProductChunk: Equatable, Sendable {
    public let range: Range<Int>
    public let text: String

    public init(range: Range<Int>, text: String) {
        self.range = range
        self.text = text
    }
}

public enum ProductChunks {
    public static func next(in text: String, from start: Int, limit: Int = 600) -> ProductChunk? {
        let scalars = Array(text.unicodeScalars)
        guard start >= 0, start < scalars.count, limit > 0 else { return nil }
        var end = min(start + limit, scalars.count)
        if end < scalars.count,
            let boundary = scalars[(start + min(limit / 2, end - start))..<end].lastIndex(where: { $0 == "\n" })
        {
            end = boundary + 1
        }
        return ProductChunk(range: start..<end, text: String(String.UnicodeScalarView(scalars[start..<end])))
    }

    public static func followingStart(after chunk: ProductChunk, overlap: Int = 64) -> Int {
        max(chunk.range.lowerBound + 1, chunk.range.upperBound - min(overlap, chunk.range.count / 4))
    }
}
