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

    @Test("VRR-02 VoxCPM2 duration bounds are checked on the source frames before conversion")
    func voxBounds() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let reader = SpeechReferenceReader(format: .voxcpm2)
        for (rate, frames, error) in [
            (16_000.0, 79_999, SpeechReferenceError.voxcpmTooShort), (16_000, 160_001, .voxcpmTooLong),
            (44_100, 220_499, .voxcpmTooShort), (44_100, 441_001, .voxcpmTooLong),
        ] {
            let url = try audio(in: root, rate: rate, frames: frames, levels: [0.25])
            await #expect(throws: error) { try await reader.read(url, source: .file("bound.wav")) }
        }
        for (rate, frames, count) in [
            (16_000.0, 160_000, 160_000), (44_100, 220_500, 80_000), (44_100, 220_501, 80_001),
        ] {
            let url = try audio(in: root, rate: rate, frames: frames, levels: [0.25, 0.25])
            let reference = try await reader.read(url, source: .file("bound.wav"))
            #expect(reference.format == .voxcpm2 && reference.samples.count == count)
            #expect(reference.samples.allSatisfy { $0.isFinite })
        }
    }

    @Test("VRR-02 Only a single missing fractional frame at a complete end of stream is completed")
    func voxTail() throws {
        var fractional = [Float](repeating: 0.25, count: 80_000)
        try SpeechReferenceReader.completeVoxResampling(
            &fractional, sourceCount: 220_501, consumedCount: 220_501, sourceRate: 44_100)
        #expect(fractional.count == 80_001 && fractional[80_000].bitPattern == 0 && fractional[79_999] == 0.25)
        var extra = [Float](repeating: 0.25, count: 80_003)
        try SpeechReferenceReader.completeVoxResampling(
            &extra, sourceCount: 220_501, consumedCount: 220_501, sourceRate: 44_100)
        #expect(extra.count == 80_001)
        for (frames, consumed, obtained) in [
            (220_501, 220_501, 79_999), (220_500, 220_500, 79_999), (220_501, 220_500, 80_000),
        ] {
            var short = [Float](repeating: 0.25, count: obtained)
            #expect(throws: SpeechReferenceError.unreadable) {
                try SpeechReferenceReader.completeVoxResampling(
                    &short, sourceCount: frames, consumedCount: consumed, sourceRate: 44_100)
            }
            #expect(short.count == obtained)
        }
    }

    @Test("VRR-03 The service reader follows the engine and the default service stays Irodori")
    @MainActor
    func serviceWiring() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = try audio(in: root, rate: 16_000, frames: 80_000, levels: [0.25])
        let vox = try await LiveSpeechService(engine: .voxcpm2, defaults: nil).importReference(url)
        #expect(vox.format == .voxcpm2 && vox.samples.count == 80_000 && vox.source == .file(url.lastPathComponent))
        let legacy = try await LiveSpeechService(defaults: nil).importReference(url)
        #expect(legacy.format == .irodori && legacy.samples.count == 240_000)
        #expect(try await SpeechReferenceReader().read(url, source: .file(url.lastPathComponent)) == legacy)
    }

    @Test("VRR-03 Cancellation surfaces unchanged and leaves the chosen file in place")
    func voxCancellation() async throws {
        let root = try directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = try audio(in: root, rate: 24_000, frames: 120_000, levels: [0.25])
        let before = try Data(contentsOf: url)
        let reader = SpeechReferenceReader(format: .voxcpm2)
        let outcome = await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await reader.read(url, source: .file("cancel.wav"))
        }.result
        #expect(throws: CancellationError.self) { try outcome.get() }
        #expect(try Data(contentsOf: url) == before)
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
