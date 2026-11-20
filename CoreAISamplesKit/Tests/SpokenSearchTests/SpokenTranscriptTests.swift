import Foundation
import SpokenSearchFeature
import Testing

struct SpokenTranscriptTests {
    @Test("Japanese words spanning individual character timestamps return each time with its own sentence")
    func japaneseOccurrences() throws {
        let resource = try #require(Bundle.module.url(forResource: "japanese-search", withExtension: "json"))
        let fixture = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: resource))
        let first = try #require(fixture["first"])
        let traffic = try #require(fixture["traffic"])
        let second = try #require(fixture["second"])
        let query = try #require(fixture["query"])
        let trafficRuns = runs(for: traffic, startingAt: 12)
        let transcript = SpokenTranscript(
            runs: runs(for: first, startingAt: 0) + trafficRuns + runs(for: second, startingAt: 30))

        let hits = transcript.matches(" \n" + query + "\t")

        #expect(hits.map(\.start) == [3, 33])
        #expect(hits.map(\.end) == [4.5, 34.5])
        let morning = try #require(hits.first)
        let afternoon = try #require(hits.last)
        #expect(morning.text.contains(first.dropLast()))
        #expect(afternoon.text.contains(second.dropLast()))
        #expect(!morning.text.contains(traffic.dropLast()))
        #expect(!afternoon.text.contains(traffic.dropLast()))
        #expect(SpokenTranscript(runs: trafficRuns).matches(query).isEmpty)
    }

    @Test("An occurrence takes its timestamps from the runs containing its first and last character")
    func spansRuns() {
        let transcript = SpokenTranscript(runs: [
            SpokenRun(text: "The for", start: 2, end: 3),
            SpokenRun(text: "", start: nil, end: nil),
            SpokenRun(text: "ec", start: 3, end: 4),
            SpokenRun(text: "ast changed.", start: 4, end: 8),
        ])
        let hits = transcript.matches("forecast")

        #expect(hits.map(\.start) == [2])
        #expect(hits.map(\.end) == [8])
        #expect(hits.first?.text.contains("The forecast changed") == true)
    }

    @Test("Repeated and overlapping words remain distinct and retain their run's actual start")
    func repeatedOccurrences() {
        let transcript = SpokenTranscript(runs: [
            SpokenRun(text: "aaaa.", start: 8, end: 13),
            SpokenRun(text: "aa", start: 2, end: 3),
            SpokenRun(text: ".", start: 3, end: 4),
            SpokenRun(text: "aa", start: 2, end: 3),
        ])
        let hits = transcript.matches("aa")

        #expect(hits.map(\.start) == [2, 2, 8, 8, 8])
        #expect(hits.map(\.end) == [3, 3, 13, 13, 13])
        #expect(Set(hits.map(\.id)).count == hits.count)
        #expect(hits == transcript.matches("aa"))
    }

    @Test("A grapheme split across runs matches its canonically equivalent spelling at both source times")
    func graphemeBoundaries() {
        let transcript = SpokenTranscript(runs: [
            SpokenRun(text: "cafe", start: 5, end: 6),
            SpokenRun(text: "\u{301}", start: 6, end: 7),
            SpokenRun(text: " \u{1F469}\u{200D}", start: 7, end: 8),
            SpokenRun(text: "\u{1F4BB}", start: 8, end: 9),
        ])
        let accent = transcript.matches("é")
        let emoji = transcript.matches("\u{1F469}\u{200D}\u{1F4BB}")

        #expect(accent.map(\.start) == [5])
        #expect(accent.map(\.end) == [7])
        #expect(emoji.map(\.start) == [7])
        #expect(emoji.map(\.end) == [9])
    }

    @Test(
        "An empty, absent or differently cased query returns no hits", arguments: ["", " \n ", "train", "BUS", "b.s"])
    func noMatch(_ query: String) {
        let transcript = SpokenTranscript(runs: [SpokenRun(text: "Take the bus", start: 0, end: 2)])

        #expect(transcript.matches(query).isEmpty)
        #expect(SpokenTranscript(runs: []).matches(query).isEmpty)
    }

    @Test("Punctuation searches literally and untimed interior spaces preserve a word's known boundaries")
    func literalPunctuationAndUntimedSpace() {
        let transcript = SpokenTranscript(runs: [
            SpokenRun(text: "[bus]", start: 1, end: 2),
            SpokenRun(text: " ", start: nil, end: nil),
            SpokenRun(text: "stop", start: 3, end: 4),
        ])
        let hits = transcript.matches("[bus] stop")

        #expect(hits.map(\.start) == [1])
        #expect(hits.map(\.end) == [4])
    }

    @Test("Missing, non-finite, negative and reversed timestamps cannot supply either end of a match")
    func invalidTimestamps() {
        let intervals: [(Double?, Double?)] = [
            (nil, 2), (1, nil), (.nan, 2), (1, .nan), (-1, 2), (2, 1), (1, .infinity), (-.infinity, 2),
        ]
        for (start, end) in intervals {
            let middle = SpokenRun(text: "y", start: start, end: end)
            let transcript = SpokenTranscript(runs: [
                SpokenRun(text: "x", start: 0, end: 1), middle, SpokenRun(text: "z", start: 2, end: 3),
            ])

            #expect(transcript.matches("y").isEmpty)
            #expect(transcript.matches("xy").isEmpty)
            #expect(transcript.matches("yz").isEmpty)
            #expect(transcript.matches("z").map(\.start) == [2])
        }
        let reversed = SpokenTranscript(runs: [
            SpokenRun(text: "x", start: 6, end: 7), SpokenRun(text: "y", start: 2, end: 3),
        ])
        #expect(reversed.matches("xy").isEmpty)
    }

    @Test("A long repeated query keeps each display excerpt short")
    func boundedExcerpt() {
        let marker = "0123456789"
        let transcript = SpokenTranscript(runs: [
            SpokenRun(text: String(repeating: marker, count: 150), start: 0, end: 10)
        ])

        let hits = transcript.matches(String(repeating: marker, count: 12))

        #expect(hits.count == 139)
        #expect(hits.allSatisfy { $0.text.count <= 80 })
    }

    private func runs(for sentence: String, startingAt start: Double) -> [SpokenRun] {
        sentence.enumerated().map { index, character in
            let onset = start + Double(index)
            return SpokenRun(text: String(character), start: onset, end: onset + 0.5)
        }
    }
}
