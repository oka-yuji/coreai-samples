import Foundation

struct ControlTokenSanitizer: Sendable {
    private let patterns: [[UInt8]]

    init(tokens: [String]) {
        patterns = tokens.filter { !$0.isEmpty }.map { Array($0.utf8) }.sorted {
            $0.count == $1.count ? $0.lexicographicallyPrecedes($1) : $0.count > $1.count
        }
    }

    func sanitized(_ text: String) -> String {
        guard !patterns.isEmpty else { return text }
        var bytes: [UInt8] = []
        bytes.reserveCapacity(text.utf8.count)
        var removed = false
        for byte in text.utf8 {
            bytes.append(byte)
            while let pattern = patterns.first(where: {
                bytes.count >= $0.count && bytes.suffix($0.count).elementsEqual($0)
            }) {
                bytes.removeLast(pattern.count)
                removed = true
            }
        }
        return removed ? String(decoding: bytes, as: UTF8.self) : text
    }
}
