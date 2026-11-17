import Foundation
import SpeechCore
import Testing

struct SpeechLimitTests {
    @Test("VRT-05 Float WAV keeps its default limit and bytes and accepts an explicit sample limit")
    func sampleLimit() throws {
        let irodoriLimit = 750 * 1920
        let samples: [Float] = [-1, 0, 0.5, 1]
        #expect(try floatWAV(samples: samples) == floatWAV(samples: samples, maximumSamples: irodoriLimit))
        #expect(try floatWAV(samples: samples) == floatWAV(samples: samples, maximumSamples: samples.count))
        let longer = [Float](repeating: 0.5, count: irodoriLimit + 1)
        #expect(throws: SpeechError.invalidSamples) { try floatWAV(samples: longer) }
        #expect(try floatWAV(samples: longer, maximumSamples: longer.count).count == 58 + longer.count * 4)
        #expect(throws: SpeechError.invalidSamples) { try floatWAV(samples: samples, maximumSamples: 3) }
        #expect(throws: SpeechError.invalidSamples) { try floatWAV(samples: [], maximumSamples: 3) }
        #expect(throws: SpeechError.invalidSamples) { try floatWAV(samples: [.nan], maximumSamples: 3) }
    }

    @Test("VoxCPM2 messages name the model and its token limit while the Irodori messages stay the same")
    func messages() {
        let voxcpm: [SpeechError] = [
            .modelNotOpened("VoxCPM2"), .textExceedsLimit(count: 145, limit: 144), .speechNotFinished, .emptyText,
            .busy, .invalidSamples, .referenceNotSupported,
        ]
        for error in voxcpm { #expect(error.errorDescription?.contains("Irodori") == false) }
        #expect(SpeechError.modelNotOpened("VoxCPM2").errorDescription == "Open a VoxCPM2 model folder first.")
        #expect(
            SpeechError.textExceedsLimit(count: 145, limit: 144).errorDescription
                == "This text contains 145 tokens. Shorten it to 144 tokens or fewer.")
        #expect(SpeechError.noModel.errorDescription == "Open an Irodori model folder first.")
        #expect(
            SpeechError.textTooLong(256).errorDescription
                == "This text contains 256 tokens. Shorten it to 255 tokens or fewer.")
    }
}
