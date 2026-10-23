import AVFoundation
import CoreGraphics
import Foundation
import Testing
import VideoSearchFeature
import VideoSearchRuntime

@Suite("Video clip export contracts", .timeLimit(.minutes(1)))
struct VideoClipExporterTests {
    @Test(
        "CLIP-03 Two seconds from 1, 7 and 12 seconds of a fifteen-second video without audio export to about six seconds whose clip edges keep the source colors"
    )
    func joinsClips() async throws {
        let colors = try (0..<15).map { try SyntheticVideo.solid(Self.hue($0, of: 15), width: 160, height: 120) }
        let source = try await SyntheticVideo.write(width: 160, height: 120, frames: 150) { colors[$0 / 10] }
        defer { SyntheticVideo.remove(source) }
        let output = source.deletingLastPathComponent().appending(path: "moments.mp4")
        let plan = VideoClipPlan.make(times: [12, 1, 7], length: 2, duration: 15)
        let seconds = try await VideoClipExporter.export(plan, of: source, to: output) { _ in }
        let asset = AVURLAsset(url: output)
        let duration = try await asset.load(.duration).seconds
        #expect(abs(seconds - 6) <= 0.01 && abs(duration - 6) <= 0.1)
        #expect(try await asset.loadTracks(withMediaType: .audio).isEmpty)
        let pairs: [(output: Double, source: Double)] = [
            (0.05, 1.05), (1.95, 2.95), (2.05, 7.05), (3.95, 8.95), (4.05, 12.05), (5.95, 13.95),
        ]
        for pair in pairs {
            let exported = try #require(SyntheticVideo.rgb(of: try await Self.frame(of: output, at: pair.output)))
            let original = try #require(SyntheticVideo.rgb(of: try await Self.frame(of: source, at: pair.source)))
            #expect(Self.matches(exported, original), "\(pair) \(exported) \(original)")
        }
        let before = try #require(SyntheticVideo.rgb(of: try await Self.frame(of: source, at: 6.95)))
        let after = try #require(SyntheticVideo.rgb(of: try await Self.frame(of: source, at: 7.05)))
        #expect(!Self.matches(before, after))
    }

    @Test("CLIP-03 A video recorded with a quarter-turn transform exports in the same portrait orientation")
    func keepsOrientation() async throws {
        let marker = try Self.halves(width: 160, height: 120)
        let source = try await SyntheticVideo.write(
            width: 160, height: 120, frames: 30, transform: CGAffineTransform(rotationAngle: .pi / 2)
        ) { _ in marker }
        defer { SyntheticVideo.remove(source) }
        let output = source.deletingLastPathComponent().appending(path: "moments.mp4")
        _ = try await VideoClipExporter.export(
            VideoClipPlan.make(times: [0.5], length: 2, duration: 3), of: source, to: output
        ) { _ in }
        let track = try #require(try await AVURLAsset(url: output).loadTracks(withMediaType: .video).first)
        let (size, transform) = try await track.load(.naturalSize, .preferredTransform)
        let shown = CGRect(origin: .zero, size: size).applying(transform)
        #expect(shown.width < shown.height)
        let exported = try await Self.frame(of: output, at: 1)
        let original = try await Self.frame(of: source, at: 1.5)
        #expect(exported.width == 120 && exported.height == 160)
        for point in [(0.5, 0.25), (0.5, 0.75)] {
            let left = try #require(SyntheticVideo.rgb(of: exported, x: point.0, y: point.1))
            let right = try #require(SyntheticVideo.rgb(of: original, x: point.0, y: point.1))
            #expect(Self.matches(left, right), "\(point) \(left) \(right)")
        }
    }

    @Test("CLIP-03 Clips before and after a short audio track export as video only")
    func clipsOutsideAudio() async throws {
        let source = try await Self.tenSecondVideo(audioFrom: 2, duration: 2)
        defer { SyntheticVideo.remove(source) }
        let audioStart = try #require(try await SyntheticVideo.audioStart(of: source))
        #expect(abs(audioStart - 2) <= 0.05)
        let output = source.deletingLastPathComponent().appending(path: "moments.mp4")
        let seconds = try await VideoClipExporter.export(
            VideoClipPlan.make(times: [0, 6], length: 1.5, duration: 10), of: source, to: output
        ) { _ in }
        let asset = AVURLAsset(url: output)
        let duration = try await asset.load(.duration).seconds
        #expect(abs(seconds - 3) <= 0.01 && abs(duration - 3) <= 0.1)
        #expect(try await asset.loadTracks(withMediaType: .audio).isEmpty)
        #expect(try await asset.loadTracks(withMediaType: .video).count == 1)
    }

    @Test("CLIP-03 A clip that overlaps part of the audio keeps the audio's delay from the start of that clip")
    func clipsPartlyInAudio() async throws {
        let source = try await Self.tenSecondVideo(audioFrom: 2, duration: 2)
        defer { SyntheticVideo.remove(source) }
        let audioStart = try #require(try await SyntheticVideo.audioStart(of: source))
        let output = source.deletingLastPathComponent().appending(path: "moments.mp4")
        let seconds = try await VideoClipExporter.export(
            VideoClipPlan.make(times: [1, 6], length: 2, duration: 10), of: source, to: output
        ) { _ in }
        #expect(abs(seconds - 4) <= 0.01)
        let heard = try #require(try await SyntheticVideo.audibleStart(of: output))
        #expect(abs(heard - (audioStart - 1)) <= 0.05, "\(heard) \(audioStart)")
    }

    @Test(
        "CLIP-03 A clip past the end of a video track shorter than its audio returns and exports only the part with video"
    )
    func clipPastVideoEnd() async throws {
        let source = try await Self.tenSecondVideo(audioFrom: 0, duration: 15)
        defer { SyntheticVideo.remove(source) }
        #expect(abs(try await AVURLAsset(url: source).load(.duration).seconds - 15) <= 0.1)
        let output = source.deletingLastPathComponent().appending(path: "moments.mp4")
        let seconds = try await VideoClipExporter.export(
            VideoClipPlan.make(times: [8.5], length: 3, duration: 15), of: source, to: output
        ) { _ in }
        #expect(abs(seconds - 1.5) <= 0.01)
        #expect(abs(try await AVURLAsset(url: output).load(.duration).seconds - 1.5) <= 0.1)
    }

    @Test("CLIP-03 Cancelling an export throws a cancellation error and leaves no file behind")
    func cancellation() async throws {
        let colors = try (0..<15).map { try SyntheticVideo.solid(Self.hue($0, of: 15), width: 160, height: 120) }
        let source = try await SyntheticVideo.write(width: 160, height: 120, frames: 150) { colors[$0 / 10] }
        defer { SyntheticVideo.remove(source) }
        let output = source.deletingLastPathComponent().appending(path: "moments.mp4")
        let plan = VideoClipPlan.make(times: [0], length: 15, duration: 15)
        let exporting = Task { try await VideoClipExporter.export(plan, of: source, to: output) { _ in } }
        exporting.cancel()
        await #expect(throws: CancellationError.self) { try await exporting.value }
        #expect(!FileManager.default.fileExists(atPath: output.path))
    }

    private static func tenSecondVideo(audioFrom start: Double, duration: Double) async throws -> URL {
        let colors = try (0..<10).map { try SyntheticVideo.solid(Self.hue($0, of: 10), width: 160, height: 120) }
        let video = try await SyntheticVideo.write(width: 160, height: 120, frames: 100) { colors[$0 / 10] }
        return try await SyntheticVideo.addingTone(to: video, at: start, duration: duration)
    }

    private static func frame(of url: URL, at seconds: Double) async throws -> CGImage {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        return try await generator.image(at: CMTime(seconds: seconds, preferredTimescale: 600)).image
    }

    private static func matches(_ left: [Int], _ right: [Int]) -> Bool {
        zip(left, right).allSatisfy { abs($0 - $1) <= 32 }
    }

    private static func hue(_ index: Int, of count: Int) -> CGColor {
        let sector = Double(index) / Double(count) * 6
        let rising = 1 - abs(sector.truncatingRemainder(dividingBy: 2) - 1)
        let (red, green, blue): (Double, Double, Double) =
            switch Int(sector) {
            case 0: (1, rising, 0)
            case 1: (rising, 1, 0)
            case 2: (0, 1, rising)
            case 3: (0, rising, 1)
            case 4: (rising, 0, 1)
            default: (1, 0, rising)
            }
        return CGColor(red: red, green: green, blue: blue, alpha: 1)
    }

    private static func halves(width: Int, height: Int) throws -> CGImage {
        guard
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw CocoaError(.fileWriteUnknown) }
        context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width / 2, height: height))
        context.setFillColor(CGColor(red: 0, green: 0, blue: 1, alpha: 1))
        context.fill(CGRect(x: width / 2, y: 0, width: width - width / 2, height: height))
        guard let image = context.makeImage() else { throw CocoaError(.fileWriteUnknown) }
        return image
    }
}

extension SyntheticVideo {
    static func addingTone(to video: URL, at start: Double, duration: Double) async throws -> URL {
        let folder = video.deletingLastPathComponent()
        let silence = folder.appending(path: "tone.caf")
        guard let format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1),
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(duration * 44_100)),
            let samples = buffer.floatChannelData?[0]
        else { throw CocoaError(.fileWriteUnknown) }
        buffer.frameLength = buffer.frameCapacity
        for index in 0..<Int(buffer.frameLength) {
            samples[index] = Float(0.5 * sin(Double(index) * 2 * .pi * 440 / 44_100))
        }
        let file = try AVAudioFile(forWriting: silence, settings: format.settings)
        try file.write(from: buffer)
        file.close()
        let output = folder.appending(path: "with-audio.mov")
        let pictureAsset = AVURLAsset(url: video)
        let soundAsset = AVURLAsset(url: silence)
        let pictures = try await pictureAsset.loadTracks(withMediaType: .video)
        let sounds = try await soundAsset.loadTracks(withMediaType: .audio)
        let movie = try AVMutableMovie(settingsFrom: nil, options: nil)
        movie.defaultMediaDataStorage = AVMediaDataStorage(url: output, options: nil)
        guard let picture = pictures.first, let sound = sounds.first,
            let videoTrack = movie.addMutableTrack(withMediaType: .video, copySettingsFrom: picture, options: nil),
            let audioTrack = movie.addMutableTrack(withMediaType: .audio, copySettingsFrom: sound, options: nil)
        else { throw CocoaError(.fileReadCorruptFile) }
        try videoTrack.insertTimeRange(
            try await picture.load(.timeRange), of: picture, at: .zero, copySampleData: true)
        try audioTrack.insertTimeRange(
            try await sound.load(.timeRange), of: sound, at: CMTime(seconds: start, preferredTimescale: 44_100),
            copySampleData: true)
        try movie.writeHeader(to: output, fileType: .mov, options: .addMovieHeaderToDestination)
        withExtendedLifetime((pictureAsset, soundAsset)) {}
        return output
    }

    static func audioStart(of url: URL) async throws -> Double? {
        let tracks = try await AVURLAsset(url: url).loadTracks(withMediaType: .audio)
        return try await tracks.first?.load(.segments).first { !$0.isEmpty }?.timeMapping.target.start.seconds
    }

    static func audibleStart(of url: URL) async throws -> Double? {
        guard let offset = try await audioStart(of: url) else { return nil }
        let file = try AVAudioFile(forReading: url)
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
        else { return nil }
        try file.read(into: buffer)
        guard let samples = buffer.floatChannelData?[0],
            let first = (0..<Int(buffer.frameLength)).first(where: { abs(samples[$0]) > 0.1 })
        else { return nil }
        return offset + Double(first) / file.processingFormat.sampleRate
    }
}
