import Foundation
import SpeechCore
import Testing

struct SpeechReferenceFormatBoundaryTests {
    @Test("VPC-01 Each format publishes its validated sample rate and inclusive limits")
    func formatLimits() throws {
        #expect(SpeechReferenceFormat.irodori.sampleRate == SpeechReferenceAudio.sampleRate)
        #expect(SpeechReferenceFormat.irodori.minimumSamples == SpeechReferenceAudio.minimumSamples)
        #expect(SpeechReferenceFormat.irodori.maximumSamples == SpeechReferenceAudio.maximumSamples)
        #expect(SpeechReferenceFormat.voxcpm2.sampleRate == 16_000)
        #expect(SpeechReferenceFormat.voxcpm2.minimumSamples == 80_000)
        #expect(SpeechReferenceFormat.voxcpm2.maximumSamples == 160_000)
        for count in [80_000, 160_000] {
            let voice = try SpeechReferenceAudio(samples: tone(count), source: .recording, format: .voxcpm2)
            #expect(voice.encoderSamples.count == 161_280)
        }
    }

    @Test("VPC-01 The legacy constructor selects Irodori and keeps reflect padding and speaker tokens")
    func legacyGeometry() throws {
        let samples = (0..<24_001).map { Float($0 % 7) / 8 }
        let legacy = try SpeechReferenceAudio(samples: samples, source: .recording)
        #expect(legacy.format == .irodori)
        #expect(legacy == (try SpeechReferenceAudio(samples: samples, source: .recording, format: .irodori)))
        #expect(legacy.frameCount == 13 && legacy.speakerTokenCount == 4 && legacy.conditioningPositionCount == 4)
        let padded = legacy.encoderSamples
        #expect(padded.count == 24_960)
        #expect(padded[24_001] == samples[23_999] && padded[24_002] == samples[23_998])
    }

    @Test("VPC-01 VoxCPM2 conditioning positions follow 2560-sample patches")
    func voxPatchBoundary() throws {
        let boundary = try SpeechReferenceAudio(samples: tone(81_920), source: .recording, format: .voxcpm2)
        let after = try SpeechReferenceAudio(samples: tone(81_921), source: .recording, format: .voxcpm2)
        #expect(boundary.frameCount == 128 && boundary.conditioningPositionCount == 32)
        #expect(after.frameCount == 132 && after.conditioningPositionCount == 33)
        #expect(after.speakerTokenCount == 33)
    }

    @Test("VPC-02 VoxCPM2 input is checked for length, then finiteness, then silence")
    func voxRejections() {
        #expect(throws: SpeechReferenceError.voxcpmTooShort) {
            try SpeechReferenceAudio(samples: [.nan], source: .recording, format: .voxcpm2)
        }
        #expect(throws: SpeechReferenceError.voxcpmTooShort) {
            try SpeechReferenceAudio(samples: tone(79_999), source: .recording, format: .voxcpm2)
        }
        #expect(throws: SpeechReferenceError.voxcpmTooLong) {
            try SpeechReferenceAudio(samples: tone(160_001), source: .recording, format: .voxcpm2)
        }
        for bad in [Float.nan, .infinity, -.infinity] {
            var samples = [Float](repeating: 0, count: 80_000)
            samples[79_999] = bad
            #expect(throws: SpeechReferenceError.invalidSamples) {
                try SpeechReferenceAudio(samples: samples, source: .recording, format: .voxcpm2)
            }
        }
        #expect(throws: SpeechReferenceError.silent) {
            try SpeechReferenceAudio(samples: [Float](repeating: -0, count: 80_000), source: .recording, format: .voxcpm2)
        }
        #expect(throws: SpeechReferenceError.tooShort) {
            try SpeechReferenceAudio(samples: tone(23_999), source: .recording, format: .irodori)
        }
    }

    @Test("VPC-02 Subnormal input keeps its bits and an over-one peak is normalized once")
    func normalization() throws {
        var quiet = [Float](repeating: -0, count: 80_000)
        quiet[7] = .leastNonzeroMagnitude
        let tiny = try SpeechReferenceAudio(samples: quiet, source: .recording, format: .voxcpm2)
        #expect(tiny.samples.map(\.bitPattern) == quiet.map(\.bitPattern))
        var loud = tone(80_000)
        loud[3] = -4
        let normalized = try SpeechReferenceAudio(samples: loud, source: .recording, format: .voxcpm2)
        #expect(normalized.samples[3] == -1 && normalized.samples[0] == 0.0625)
        let legacy = try SpeechReferenceAudio(samples: tiny.samples + quiet, source: .recording)
        #expect(legacy.seconds == Double(160_000) / 48_000 && tiny.seconds == 5)
        #expect(legacy != (try SpeechReferenceAudio(samples: tiny.samples + quiet, source: .recording, format: .voxcpm2)))
    }

    @Test("VPC-03 Reference previews write complete mono Float32 RIFF headers at each format rate")
    func previewHeaders() throws {
        let vox = try SpeechReferenceAudio(samples: tone(80_000), source: .recording, format: .voxcpm2)
        let data = try floatWAV(reference: vox)
        #expect(data.count == 58 + 80_000 * 4)
        #expect(word(data, 4) == UInt32(50 + 80_000 * 4))
        #expect(word(data, 24) == 16_000 && word(data, 28) == 64_000)
        #expect(word(data, 46) == 80_000 && word(data, 54) == UInt32(80_000 * 4))
        #expect(word(data, 58) == Float(0.25).bitPattern)
        let legacy = try SpeechReferenceAudio(samples: tone(24_000), source: .recording)
        let legacyData = try floatWAV(reference: legacy)
        #expect(legacyData == (try floatWAV(samples: legacy.samples)))
        #expect(word(legacyData, 24) == 48_000 && word(legacyData, 28) == 192_000)
    }

    private func tone(_ count: Int) -> [Float] { [Float](repeating: 0.25, count: count) }

    private func word(_ data: Data, _ offset: Int) -> UInt32 {
        data[offset..<offset + 4].enumerated().reduce(UInt32(0)) { $0 | UInt32($1.element) << (8 * $1.offset) }
    }
}
