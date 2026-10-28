import SpeechCore
import Testing

struct SpeechReferenceAudioTests {
    @Test("REF-CORE-01 Reference audio rejects empty short long nonfinite and silent input")
    func invalidInput() {
        for count in [0, 23_999] {
            #expect(throws: SpeechReferenceError.tooShort) {
                try SpeechReferenceAudio(samples: [Float](repeating: 0.5, count: count), source: .recording)
            }
        }
        #expect(throws: SpeechReferenceError.tooLong) {
            try SpeechReferenceAudio(samples: [Float](repeating: 0.5, count: 1_440_001), source: .recording)
        }
        #expect(throws: SpeechReferenceError.silent) {
            try SpeechReferenceAudio(samples: [Float](repeating: 0, count: 24_000), source: .recording)
        }
        for invalid in [Float.nan, .infinity, -.infinity] {
            var samples = [Float](repeating: 0.5, count: 24_000)
            samples[12] = invalid
            #expect(throws: SpeechReferenceError.invalidSamples) {
                try SpeechReferenceAudio(samples: samples, source: .recording)
            }
        }
    }

    @Test("REF-CORE-01 Audio boundaries use reflection and discard incomplete speaker patches")
    func boundaries() throws {
        var samples = [Float](repeating: 0.5, count: 24_001)
        samples.replaceSubrange(23_997...24_000, with: [0.1, 0.2, 0.3, 0.4])
        let reference = try SpeechReferenceAudio(samples: samples, source: .file("voice.wav"))
        #expect(reference.frameCount == 13)
        #expect(reference.speakerTokenCount == 4)
        let padded = reference.encoderSamples
        #expect(padded.count == 24_960)
        #expect(Array(padded[24_000...24_004]) == [0.4, 0.3, 0.2, 0.1, 0.5])
        #expect(Array(padded.prefix(samples.count)) == samples)
        #expect(reference.source == .file("voice.wav"))
        let longest = try SpeechReferenceAudio(
            samples: [Float](repeating: 0.25, count: 1_440_000), source: .recording)
        #expect(longest.seconds == 30)
        #expect(longest.frameCount == 750 && longest.speakerTokenCount == 188)
        #expect(longest.encoderSamples == longest.samples)
    }

    @Test("REF-CORE-01 Peak normalization scales the entire signal only when its peak exceeds one")
    func scaling() throws {
        var samples = [Float](repeating: 0.5, count: 24_000)
        samples[0] = -2
        let reference = try SpeechReferenceAudio(samples: samples, source: .recording)
        #expect(reference.samples[0] == -1)
        #expect(reference.samples[1] == 0.25)
        let repeated = try SpeechReferenceAudio(samples: reference.samples, source: .recording)
        #expect(repeated.samples == reference.samples)
    }
}
