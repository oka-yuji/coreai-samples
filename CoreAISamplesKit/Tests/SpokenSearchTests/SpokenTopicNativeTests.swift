import Foundation
import SpokenSearchFeature
import Testing

@testable import SpokenSearchRuntime

@MainActor
struct SpokenTopicNativeTests {
    @Test(
        "A Japanese question finds the passage advising an umbrella and plays from its start",
        .enabled(
            if: ProcessInfo.processInfo.environment["COREAI_SPOKEN_TOPIC_AUDIO"] != nil
                && ProcessInfo.processInfo.environment["COREAI_SPOKEN_TOPIC_MODEL"] != nil))
    func relatedPassage() async throws {
        let env = ProcessInfo.processInfo.environment
        let url = URL(filePath: try #require(env["COREAI_SPOKEN_TOPIC_AUDIO"]))
        let folder = URL(filePath: try #require(env["COREAI_SPOKEN_TOPIC_MODEL"]))
        let fixture = try #require(Bundle.module.url(forResource: "topic-search", withExtension: "json"))
        let text = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: fixture))
        let query = try #require(text["query"])
        let anchor = try #require(text["anchor"])
        let expected = try #require(text["expected_first"])
        let transcript = try await SpokenFileTranscriber().transcribe(url)
        #expect(transcript.matches(query).isEmpty)
        let literal = try #require(transcript.matches(anchor).first)
        let index = SpokenTopicIndex(folder: folder)
        let matches = try await index.search(query, transcript: transcript, generation: 1)
        try #require(matches.count >= 3)
        let first = try #require(matches.first)
        #expect(first.text == expected)
        #expect(first.start <= literal.start && literal.start < first.end)
        var began: Double?
        let player = SpokenRecordingPlayer(muted: true)
        try await player.play(url, from: first.start) { began = $0 }
        let observed = try #require(began)
        #expect(abs(observed - first.start) < 0.02)
        print("Ranked", matches.count, "passages; first:", first.text, "at", first.start, "played from", observed)
    }
}
