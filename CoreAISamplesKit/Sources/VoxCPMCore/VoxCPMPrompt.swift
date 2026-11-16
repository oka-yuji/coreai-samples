public enum VoxCPMPrompt {
    private static let audioStartTokenID: Int32 = 101
    private static let wordBoundary: Unicode.Scalar = "\u{2581}"
    private static let kanji: ClosedRange<Unicode.Scalar> = "\u{4E00}"..."\u{9FFF}"

    public static func inputIDs(textTokenIDs: [Int32], vocabulary: [Int32: String]) throws -> [Int32] {
        let kanjiIDs = kanjiTokenIDs(in: vocabulary)
        var ids: [Int32] = []
        for id in textTokenIDs {
            guard let token = vocabulary[id] else { throw VoxCPMError.unknownToken }
            let characters: [Unicode.Scalar] = token.unicodeScalars.filter { $0 != wordBoundary }
            guard characters.count >= 2, kanjiIDs[characters] != nil else {
                ids.append(id)
                continue
            }
            for character in characters {
                guard let characterID = kanjiIDs[[character]] else { throw VoxCPMError.unknownToken }
                ids.append(characterID)
            }
        }
        return ids + [audioStartTokenID]
    }

    private static func kanjiTokenIDs(in vocabulary: [Int32: String]) -> [[Unicode.Scalar]: Int32] {
        var ids: [[Unicode.Scalar]: Int32] = [:]
        for (id, token) in vocabulary where !token.isEmpty && token.unicodeScalars.allSatisfy(kanji.contains) {
            let scalars = Array(token.unicodeScalars)
            ids[scalars] = min(id, ids[scalars] ?? id)
        }
        return ids
    }
}
