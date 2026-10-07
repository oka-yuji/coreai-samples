/// A correction word found in the text by string match. It only marks the place: which value stands is for people to
/// decide.
public struct CorrectionCue: Equatable, Sendable {
    /// Range of the word in the text, in Unicode scalars.
    public var range: Range<Int>
    public var word: String

    public init(range: Range<Int>, word: String) {
        self.range = range
        self.word = word
    }

    /// The words of `words` in `text`, left to right and without overlap. Where several start at the same place, the
    /// longest is taken.
    public static func find(in text: String, words: [String]) -> [CorrectionCue] {
        let scalars = Array(text.unicodeScalars)
        let longestFirst = words.filter { !$0.isEmpty }.sorted { $0.unicodeScalars.count > $1.unicodeScalars.count }
        var cues: [CorrectionCue] = []
        var index = 0
        while index < scalars.count {
            guard let word = longestFirst.first(where: { scalars[index...].starts(with: $0.unicodeScalars) }) else {
                index += 1
                continue
            }
            let end = index + word.unicodeScalars.count
            cues.append(CorrectionCue(range: index..<end, word: word))
            index = end
        }
        return cues
    }
}
