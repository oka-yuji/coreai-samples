import Foundation
import SpokenSearchFeature
import Testing

@testable import SpokenSearchRuntime

struct SpokenTopicTests {
    @Test("Questions rank timed passages by similarity and break ties in recording order")
    func ranksAtOriginalTimes() async throws {
        let model = SpokenModelProbe()
        let index = SpokenTopicIndex(folder: URL(filePath: "/model")) { model }
        let transcript = SpokenTranscript(runs: [
            SpokenRun(text: "Train.", start: 0.1, end: 0.8),
            SpokenRun(text: "Wet.", start: 4, end: 4.5),
            SpokenRun(text: "Rain.", start: 7, end: 8),
        ])
        let matches = try await index.search("Should I carry a coat?", transcript: transcript, generation: 1)
        #expect(matches.map(\.text) == ["Wet.", "Rain.", "Train."])
        #expect(matches.map(\.start) == [4, 7, 0.1])
        #expect(await model.queries == ["Should I carry a coat?"])
        #expect(await model.documents.count == 3)
        let again = try await index.search("Do I need cover?", transcript: transcript, generation: 1)
        #expect(again == matches)
        #expect(await model.documents.count == 3)
        #expect(await model.loads == 2)
        #expect(await model.unloads == 2)
    }

    @Test("A replaced recording cannot reuse its passages or return a superseded result")
    func invalidatesOldRecording() async throws {
        let model = SpokenModelProbe()
        let index = SpokenTopicIndex(folder: URL(filePath: "/model")) { model }
        let old = SpokenTranscript(runs: [SpokenRun(text: "Train", start: 1, end: 2)])
        let new = SpokenTranscript(runs: [SpokenRun(text: "Rain", start: 5, end: 6)])
        _ = try await index.search("Cover", transcript: old, generation: 1)
        await index.invalidate(before: 2)
        await #expect(throws: CancellationError.self) {
            try await index.search("Cover", transcript: old, generation: 1)
        }
        let result = try await index.search("Cover", transcript: new, generation: 2)
        #expect(result.map(\.text) == ["Rain"] && result.map(\.start) == [5])
        #expect(await model.documents == ["Train", "Rain"])
    }

    @Test("Long passages split at measured positions and silent or untimed passages yield no playable hit")
    func longPassages() async throws {
        let model = SpokenModelProbe(characterLimit: 4)
        let index = SpokenTopicIndex(folder: URL(filePath: "/model")) { model }
        let transcript = SpokenTranscript(runs: [
            SpokenRun(text: "!", start: nil, end: nil),
            SpokenRun(text: "rail", start: 0, end: 1),
            SpokenRun(text: "rain", start: 1, end: 2),
        ])
        let hits = try await index.search("coat", transcript: transcript, generation: 0)
        #expect(hits.map(\.text) == ["rain", "rail"])
        #expect(hits.map(\.start) == [1, 0])
        let empty = SpokenTranscript(runs: [SpokenRun(text: "rain", start: nil, end: nil)])
        let none = try await index.search("coat", transcript: empty, generation: 1)
        #expect(none.isEmpty)
    }
}
