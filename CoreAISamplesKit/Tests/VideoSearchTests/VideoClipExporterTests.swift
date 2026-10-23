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
        try await VideoClipExporter.export(plan, of: source, to: output) { _ in }
        let asset = AVURLAsset(url: output)
        #expect(abs(try await asset.load(.duration).seconds - 6) <= 0.1)
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
        try await VideoClipExporter.export(
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
