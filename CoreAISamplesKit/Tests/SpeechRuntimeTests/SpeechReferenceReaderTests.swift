import AVFAudio
import Foundation
import SpeechCore
import Testing

@testable import SpeechRuntime

@Suite(.timeLimit(.minutes(1)))
struct SpeechReferenceReaderTests {
    @Test("REF-UI-03 Stereo file input is averaged into a mono reference without changing the source")
    func stereo() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = try audio(in: root, rate: 48_000, frames: 48_000, levels: [0.75, -0.25])
        let before = try Data(contentsOf: url)
        let reference = try await SpeechReferenceReader().read(url, source: .file("stereo.wav"))
        #expect(reference.samples.count == 48_000)
        #expect(reference.samples.allSatisfy { abs($0 - 0.25) < 0.0001 })
        #expect(reference.source == .file("stereo.wav"))
        #expect(try Data(contentsOf: url) == before)
    }

    @Test(
        "REF-UI-03 Supported sample rates convert to a finite 48 kHz reference",
        arguments: [24_000.0, 44_100.0, 96_000.0])
    func resampling(rate: Double) async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = try audio(in: root, rate: rate, frames: Int(rate), levels: [0.25])
        let reference = try await SpeechReferenceReader().read(url, source: .file("rate.wav"))
        #expect(reference.samples.count == 48_000)
        #expect(reference.samples.allSatisfy { $0.isFinite })
        #expect(abs(reference.samples[24_000] - 0.25) < 0.001)
    }

    @Test("REF-UI-03 Short long silent and unreadable audio fail before generation")
    func invalid() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let reader = SpeechReferenceReader()
        let short = try audio(in: root, rate: 48_000, frames: 12_000, levels: [0.25])
        await #expect(throws: SpeechReferenceError.tooShort) {
            try await reader.read(short, source: .file("short.wav"))
        }
        let long = try audio(in: root, rate: 48_000, frames: 1_440_001, levels: [0.25])
        await #expect(throws: SpeechReferenceError.tooLong) { try await reader.read(long, source: .file("long.wav")) }
        let silent = try audio(in: root, rate: 48_000, frames: 24_000, levels: [0])
        await #expect(throws: SpeechReferenceError.silent) { try await reader.read(silent, source: .recording) }
        let broken = root.appending(path: "broken.wav")
        try Data("not audio".utf8).write(to: broken)
        await #expect(throws: SpeechReferenceError.unreadable) {
            try await reader.read(broken, source: .file("broken.wav"))
        }
    }

    @Test("REF-UI-03 The half-second resampling boundary remains a valid reference")
    func minimumResampling() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = try audio(in: root, rate: 24_000, frames: 12_000, levels: [0.25])
        let reference = try await SpeechReferenceReader().read(url, source: .recording)
        #expect(reference.samples.count == 24_000)
    }

    private func directory() throws -> URL {
        let root = URL.temporaryDirectory.appending(path: "reference-reader-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func audio(in root: URL, rate: Double, frames: Int, levels: [Float]) throws -> URL {
        let url = root.appending(path: "\(UUID().uuidString).wav")
        let format = try #require(
            AVAudioFormat(standardFormatWithSampleRate: rate, channels: AVAudioChannelCount(levels.count)))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)))
        buffer.frameLength = AVAudioFrameCount(frames)
        let pointers = try #require(buffer.floatChannelData)
        for (channel, level) in levels.enumerated() {
            for frame in 0..<frames { pointers[channel][frame] = level }
        }
        let file = try AVAudioFile(
            forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        try file.write(from: buffer)
        return url
    }
}
