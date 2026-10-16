import Foundation

public struct IrodoriTextInput: Sendable, Equatable {
    public let normalizedText: String
    public let originalTokenCount: Int
    public let tokenIDs: [Int32]
    public let attentionMask: [Bool]

    public init(normalizedText: String, bodyTokenIDs: [UInt32], originalTokenCount: Int? = nil) throws {
        guard !normalizedText.isEmpty else { throw IrodoriInputError.emptyText }
        self.normalizedText = normalizedText
        let count = originalTokenCount ?? bodyTokenIDs.count
        guard count >= bodyTokenIDs.count else { throw IrodoriInputError.invalidTokenID }
        self.originalTokenCount = count
        var ids: [Int32] = [1]
        for token in bodyTokenIDs.prefix(255) {
            guard token < 102_400, let id = Int32(exactly: token) else { throw IrodoriInputError.invalidTokenID }
            ids.append(id)
        }
        let valid = ids.count
        ids.append(contentsOf: repeatElement(3, count: 256 - valid))
        tokenIDs = ids
        attentionMask = (0..<256).map { $0 < valid }
    }
}
