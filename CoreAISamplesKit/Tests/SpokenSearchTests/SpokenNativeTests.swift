import Foundation
import Speech
import SpokenSearchFeature
import Testing

@testable import SpokenSearchRuntime

@MainActor
struct SpokenNativeTests {
    @Test("An installed equivalent locale is usable even when the inventory reports only supported")
    func installedLanguage() {
        let japanese = Locale(identifier: "ja-JP")
        let installed = [Locale(identifier: "ja_JP")]
        #expect(
            SpokenFileTranscriberJob.effectiveStatus(.supported, for: japanese, installedLocales: installed)
                == .installed)
        #expect(SpokenFileTranscriberJob.effectiveStatus(.supported, for: japanese, installedLocales: []) == .supported)
        #expect(SpokenFileTranscriberJob.effectiveStatus(.installed, for: japanese, installedLocales: []) == .installed)
    }

    @Test("The transcriber rejects unreadable audio files before requesting speech assets")
    func unreadableAudio() async throws {
        let url = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString + ".wav")
        try Data([0, 1, 2]).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let transcriber = SpokenFileTranscriber()
        await #expect(throws: SpokenRecognitionError.unreadableFile) { try await transcriber.transcribe(url) }
        await transcriber.cancel()
    }

    @Test("Cancellation joins a slow speech operation before another recording can start", arguments: [false, true])
    func cancellationWaits(cancelCaller: Bool) async throws {
        let transcriber = SpokenFileTranscriber()
        let (starts, started) = AsyncStream.makeStream(of: CheckedContinuation<Void, Never>.self)
        let (stops, finishing) = AsyncStream<Never>.makeStream()
        let reading = Task {
            try await transcriber.run {
                await withCheckedContinuation { release in
                    started.yield(release)
                    started.finish()
                }
                return SpokenTranscript(runs: [])
            } finish: {
                finishing.finish()
            }
        }
        var start = starts.makeAsyncIterator()
        let release = try #require(await start.next())
        var returned = false
        let stopping = Task {
            if cancelCaller {
                reading.cancel()
                _ = await reading.result
            } else {
                await transcriber.cancel()
            }
            returned = true
        }
        var stop = stops.makeAsyncIterator()
        _ = await stop.next()
        #expect(!returned)
        await #expect(throws: SpokenRecognitionError.busy) {
            try await transcriber.transcribe(URL(filePath: "unused.wav"))
        }
        release.resume()
        await stopping.value
        await #expect(throws: CancellationError.self) { try await reading.value }
        let next = try await transcriber.run(recognize: { SpokenTranscript(runs: []) }, finish: {})
        #expect(next.runs.isEmpty)
    }

    @Test(
        "Two spoken occurrences play from their own positions and a related recording produces no matches",
        .enabled(if: ProcessInfo.processInfo.environment["COREAI_SPOKEN_WORD_AUDIO"] != nil))
    func fixedRecordings() async throws {
        let audio = URL(filePath: try #require(ProcessInfo.processInfo.environment["COREAI_SPOKEN_WORD_AUDIO"]))
        let related = audio.deletingLastPathComponent().appending(path: "related.wav")
        let resource = try #require(Bundle.module.url(forResource: "japanese-search", withExtension: "json"))
        let strings = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: resource))
        let query = try #require(strings["query"])
        let transcriber = SpokenFileTranscriber()
        let player = SpokenRecordingPlayer(muted: true)
        defer { player.stop() }
        let transcript = try await transcriber.transcribe(audio)
        let hits = transcript.matches(query)
        try #require(hits.count == 2)
        let expected = [(3.06, 3.36), (4.86, 5.16)]
        for (hit, range) in zip(hits, expected) {
            #expect(abs(hit.start - range.0) < 0.12)
            #expect(abs(hit.end - range.1) < 0.12)
            var started: Double?
            try await player.play(audio, from: hit.start) { started = $0 }
            let actual = try #require(started)
            #expect(abs(actual - hit.start) < 0.01)
            print("Spoken occurrence from \(hit.start) to \(hit.end) played from \(actual)")
        }
        let other = try await transcriber.transcribe(related)
        try #require(other.runs.contains { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })
        #expect(other.matches(query).isEmpty)
        await transcriber.cancel()
    }
}
