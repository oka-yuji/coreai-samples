import SpeechCore
import Testing

struct SpeechReferenceFormatTests {
    @Test("VPC-01 A five-second VoxCPM2 reference keeps the original PCM and pads only the encoder input")
    func fiveSecondReference() throws {
        var samples = [Float](repeating: 0.25, count: 80_000)
        samples[0] = -2
        let voice = try SpeechReferenceAudio(samples: samples, source: .recording, format: .voxcpm2)

        #expect(voice.format == .voxcpm2)
        #expect(voice.seconds == 5)
        #expect(voice.samples.count == 80_000)
        #expect(voice.samples[0] == -1 && voice.samples[1] == 0.125)
        #expect(voice.frameCount == 128 && voice.conditioningPositionCount == 32)
        let padded = voice.encoderSamples
        #expect(padded.count == 161_280)
        #expect(padded.prefix(80_000).elementsEqual(voice.samples))
        #expect(padded.suffix(from: 80_000).allSatisfy { $0.bitPattern == 0 })
    }
}
