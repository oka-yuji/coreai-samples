import AVFAudio
import Foundation
import SpeechCore
import Testing

@testable import SpeechRuntime

struct SpeechReferenceMixingTests {
    @Test("VRR-01 Stereo mixing keeps the legacy positive-zero accumulator in both reference formats")
    func signedZero() async throws {
        let directory = URL.temporaryDirectory.appending(path: "reference-mix-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        for kind in [SpeechReferenceFormat.irodori, .voxcpm2] {
            let count = kind.minimumSamples
            let rate = Double(kind.sampleRate)
            let format = try #require(AVAudioFormat(standardFormatWithSampleRate: rate, channels: 2))
            let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)))
            buffer.frameLength = AVAudioFrameCount(count)
            let channels = try #require(buffer.floatChannelData)
            for channel in 0..<2 {
                for frame in 0..<count { channels[channel][frame] = 0.25 }
                channels[channel][0] = Float(bitPattern: 0x8000_0000)
            }
            let url = directory.appending(path: "\(kind.sampleRate).wav")
            do {
                let file = try AVAudioFile(forWriting: url, settings: format.settings)
                try file.write(from: buffer)
            }
            let reference = try await SpeechReferenceReader(format: kind).read(url, source: .recording)
            var expected = [Float](repeating: 0.25, count: count)
            expected[0] = 0
            #expect(reference.samples.map(\.bitPattern) == expected.map(\.bitPattern))
            if kind == .irodori {
                #expect(try floatWAV(reference: reference) == floatWAV(samples: expected))
            }
        }
    }
}
