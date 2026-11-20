import AVFAudio
import Foundation
import SpeechCore
import Testing

@testable import SpeechRuntime

struct SpeechVoxReferenceReaderTests {
    @Test("VRR-01 The VoxCPM2 reader preserves 16 kHz mono PCM including signed zero")
    func directPCM() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(path: "input.wav")
        let format = try #require(AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 80_000))
        buffer.frameLength = 80_000
        let pcm = try #require(buffer.floatChannelData?[0])
        for index in 0..<80_000 { pcm[index] = 0.125 }
        pcm[1] = Float(bitPattern: 0x8000_0000)
        do {
            let file = try AVAudioFile(forWriting: url, settings: format.settings)
            try file.write(from: buffer)
        }
        let reference = try await SpeechReferenceReader(format: .voxcpm2).read(url, source: .file("input.wav"))
        #expect(reference.format == .voxcpm2 && reference.seconds == 5)
        #expect(reference.samples.count == 80_000)
        #expect(reference.samples[0] == 0.125 && reference.samples[1].bitPattern == 0x8000_0000)
    }
}
